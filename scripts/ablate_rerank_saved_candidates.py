"""在冻结的 NanoBEIR top-10 文档候选内离线验证 cross-encoder 排序。

不重新计算 corpus/query embedding，不连接数据库。候选来自已冻结的
``nanobeir-global-bm25.json``；对候选原文重新执行当前 child chunking，CE 对 chunk
打分后聚合回 document ranking。

这只能回答“CE 能否改善已有 top-10 候选的前五排序”，不能回答 40 候选池里能否
救回第 11–40 名文档。输出会永久记录这个边界，不能和完整检索评测混称。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from llama_index.core.schema import NodeWithScore  # noqa: E402

from ir_metrics import recall_at_k, summarize  # noqa: E402
from nanobeir_data import (  # noqa: E402
    SPECS, aggregate_document_ranking, document_filename, download_and_load,
)
from minibrain.config import get_config  # noqa: E402
from minibrain.modules.vector_rag import chain  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "eval" / "results" / "nanobeir-global-bm25.json"
DEFAULT_OUTPUT = ROOT / "eval" / "results" / "nanobeir-ce-saved-top10.json"
DEFAULT_CACHE = ROOT / "eval" / "cache" / "nanobeir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def _candidate_nodes(dataset, ranking: list[str]) -> tuple[list[NodeWithScore], dict[str, str]]:
    by_id = {document.doc_id: document for document in dataset.corpus}
    filename_to_doc_id = {}
    nodes = []
    for doc_id in ranking[:10]:
        document = by_id[doc_id]
        filename = document_filename(dataset.spec.name, doc_id)
        filename_to_doc_id[filename] = doc_id
        for node in chain._prepare_nodes(
            [(filename, document.content)], source_name="offline/nanobeir",
            visibility="private", owner_id="offline",
        ):
            nodes.append(NodeWithScore(node=node, score=0.0))
    return nodes, filename_to_doc_id


def _win_loss(required: set[str], before: list[str], after: list[str], k: int) -> int:
    old = recall_at_k(required, before, k)
    new = recall_at_k(required, after, k)
    return (new > old) - (new < old)


def run_task(dataset, frozen_task: dict) -> dict:
    baseline_rankings = frozen_task["arms"]["hybrid"]["rankings"]
    query_ids = list(dataset.queries)
    # 冷启动单独发生，不混进逐题在线延迟。
    chain.make_cross_encoder_rerank(1)

    baseline_pairs = []
    reranked_pairs = []
    reranked_rankings = {}
    latencies = []
    wins = {3: [0, 0, 0], 5: [0, 0, 0], 10: [0, 0, 0]}
    for query_id in query_ids:
        baseline = baseline_rankings[query_id][:10]
        nodes, filename_to_doc_id = _candidate_nodes(dataset, baseline)
        started = time.perf_counter()
        reranked = chain.make_cross_encoder_rerank(len(nodes)).postprocess_nodes(
            nodes, query_str=dataset.queries[query_id])
        latencies.append((time.perf_counter() - started) * 1000)
        ranking = aggregate_document_ranking(
            (f"{item.node.metadata['filename']} #0" for item in reranked),
            filename_to_doc_id,
        )
        required = set(dataset.qrels[query_id])
        baseline_pairs.append((required, baseline))
        reranked_pairs.append((required, ranking))
        reranked_rankings[query_id] = ranking
        for k in wins:
            outcome = _win_loss(required, baseline, ranking, k)
            wins[k][0 if outcome > 0 else 1 if outcome < 0 else 2] += 1

    ordered = sorted(latencies)
    return {
        "queries": len(query_ids),
        "candidate_source": "frozen hybrid top-10 document rankings",
        "candidate_limit": 10,
        "rerank_unit": "current child chunks; max-ranked chunk determines document rank",
        "model_cold_start_excluded": True,
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": ordered[max(0, int(len(ordered) * 0.95 + 0.999999) - 1)],
        "wins_losses_ties_by_recall": {
            f"@{k}": {"wins": row[0], "losses": row[1], "ties": row[2]}
            for k, row in wins.items()
        },
        "baseline": {"metrics": summarize(baseline_pairs, (3, 5, 10)),
                     "rankings": baseline_rankings},
        "ce": {"metrics": summarize(reranked_pairs, (3, 5, 10)),
               "rankings": reranked_rankings},
    }


def main() -> int:
    # 这条脚本的定义就是“使用已冻结候选和已缓存模型，不访问网络”。在第一次
    # dataset/CrossEncoder 加载之前设置；模块被测试 import 时不污染全局环境。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    frozen = json.loads(args.input.read_text(encoding="utf-8"))
    cfg = get_config()
    output = {
        "benchmark": "NanoBEIR frozen top-10 candidate rerank (not full retrieval)",
        "source_result": str(args.input.relative_to(ROOT)),
        "limitation": (
            "CE cannot introduce a document outside the frozen top-10; this is not the "
            "40-candidate end-to-end run."
        ),
        "reranker_model": cfg.rerank_ce_model,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "tasks": {},
    }
    for task_name, spec in SPECS.items():
        print(f"{task_name}: 加载冻结候选", flush=True)
        dataset = download_and_load(spec, args.cache_dir)
        output["tasks"][task_name] = run_task(dataset, frozen["tasks"][task_name])
        task = output["tasks"][task_name]
        b, c = task["baseline"]["metrics"], task["ce"]["metrics"]
        print(
            f"  Complete Recall@5 {b['complete_recall@5']:.3f} → "
            f"{c['complete_recall@5']:.3f} | NDCG@5 {b['ndcg@5']:.3f} → "
            f"{c['ndcg@5']:.3f} | latency p50 {task['latency_median_ms']:.0f}ms",
            flush=True,
        )
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
