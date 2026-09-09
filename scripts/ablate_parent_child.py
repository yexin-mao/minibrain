"""flat child 与 small-to-big parent 的受控消融。

每题只执行一次 child 检索；三组共享字节级相同的候选排名，依次拆分
parent-aware 去重与 ``expand_parent``。随后都通过生产的 ContextAssembler（4000 token）。

运行：uv run --no-sync python scripts/ablate_parent_child.py
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time
import uuid

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ir_metrics import summarize                                      # noqa: E402
from minibrain import identity                                        # noqa: E402
from minibrain.agent.context import ContextAssembler                   # noqa: E402
from minibrain.config import get_config                               # noqa: E402
from minibrain.db import close_all                                    # noqa: E402
from minibrain.modules.vector_rag import chain                        # noqa: E402
from minibrain.modules.vector_rag.selection import deduplicate_nodes  # noqa: E402
from minibrain.scripts_purge import purge_user                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
PROBES = ROOT / "eval" / "probes.json"
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"
OUT_JSON = ROOT / "eval" / "results" / "ablate_parent_child.json"
OUT_MD = ROOT / "eval" / "results" / "ablate_parent_child.md"
TOP_K = 12
FETCH_K = TOP_K * 2


def _files(evidence) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in evidence:
        filename = item.location.split(" #", 1)[0]
        if filename not in seen:
            seen.add(filename)
            out.append(filename)
    return out


def _arm_row(probe: dict, evidence: list, projection_ms: float) -> dict:
    cfg = get_config()
    for index, item in enumerate(evidence, start=1):
        item.evidence_id = f"E1.{index}"
    assembler = ContextAssembler(
        token_budget=cfg.agent_evidence_token_budget,
        evidence_limit=cfg.agent_evidence_limit,
        candidate_pool=cfg.agent_context_candidate_pool,
    )
    selected = assembler.add(evidence)
    metrics = assembler.metrics()
    files = _files(selected)
    expected = [str(value) for value in probe.get("expect_answer", [])]
    context = "\n\n".join(item.snippet for item in selected)
    expected_hits = sum(value.casefold() in context.casefold() for value in expected)
    return {
        "files": files,
        "candidate_count": len(evidence),
        "selected_count": len(selected),
        "unique_documents": len(files),
        "same_document_slot_rate": (
            1 - len(files) / len(selected) if selected else 0.0),
        "context_chars": sum(len(item.snippet) for item in selected),
        "context_tokens": metrics.context_tokens,
        "budget_drops": metrics.dropped_budget_count,
        "duplicate_drops": metrics.dropped_duplicate_count,
        "projection_ms": projection_ms,
        "expected_hits": expected_hits,
        "expected_total": len(expected),
        "all_expected_present": expected_hits == len(expected) if expected else None,
    }


def _aggregate(probes: list[dict], rows: list[dict]) -> dict:
    pairs = [(set(probe["required"]), row["files"])
             for probe, row in zip(probes, rows)]
    strict_pairs = [(required, ranking) for probe, (required, ranking)
                    in zip(probes, pairs) if probe.get("complete_required", True)]
    ir = summarize(pairs, (3, 5, 10))
    strict = summarize(strict_pairs, (3, 5, 10))
    expected_rows = [row for row in rows if row["all_expected_present"] is not None]
    return {
        **ir,
        "strict_queries": len(strict_pairs),
        "strict_complete_recall@5": strict.get("complete_recall@5"),
        "strict_complete_recall@10": strict.get("complete_recall@10"),
        "expected_answer_queries": len(expected_rows),
        "all_expected_present_rate": (
            sum(bool(row["all_expected_present"]) for row in expected_rows)
            / len(expected_rows) if expected_rows else None),
        "selected_count_median": statistics.median(
            row["selected_count"] for row in rows),
        "unique_documents_median": statistics.median(
            row["unique_documents"] for row in rows),
        "same_document_slot_rate_mean": statistics.mean(
            row["same_document_slot_rate"] for row in rows),
        "context_chars_median": statistics.median(row["context_chars"] for row in rows),
        "context_tokens_median": statistics.median(row["context_tokens"] for row in rows),
        "context_tokens_p95": sorted(row["context_tokens"] for row in rows)[
            max(0, int(len(rows) * 0.95) - 1)],
        "queries_with_budget_drops": sum(row["budget_drops"] > 0 for row in rows),
        "duplicate_drops": sum(row["duplicate_drops"] for row in rows),
        "projection_latency_median_ms": statistics.median(
            row["projection_ms"] for row in rows),
    }


def _markdown(payload: dict) -> str:
    flat = payload["arms"]["flat_child"]
    dedup = payload["arms"]["child_parent_dedup"]
    parent = payload["arms"]["small_to_big"]
    delta_tokens = parent["context_tokens_median"] - flat["context_tokens_median"]
    delta_support = ((parent["all_expected_present_rate"] or 0)
                     - (flat["all_expected_present_rate"] or 0))
    expansion_recall = parent["recall@5"] - dedup["recall@5"]
    expansion_support = ((parent["all_expected_present_rate"] or 0)
                         - (dedup["all_expected_present_rate"] or 0))
    return f"""# 父子检索消融

同一份 800 字 child 索引、同一批 child 排名；三组依次拆分 parent-aware 去重与
1600 字 parent 正文展开。三组最终都经过生产的 4000-token ContextAssembler，
没有调用生成式 LLM。

| 指标 | flat child | child + parent 去重 | small-to-big |
|---|---:|---:|---:|
| Recall@5 | {flat['recall@5']:.3f} | {dedup['recall@5']:.3f} | {parent['recall@5']:.3f} |
| NDCG@5 | {flat['ndcg@5']:.3f} | {dedup['ndcg@5']:.3f} | {parent['ndcg@5']:.3f} |
| MRR | {flat['mrr']:.3f} | {dedup['mrr']:.3f} | {parent['mrr']:.3f} |
| 严格 Complete Recall@5 | {flat['strict_complete_recall@5']:.3f} | {dedup['strict_complete_recall@5']:.3f} | {parent['strict_complete_recall@5']:.3f} |
| 关键答案字符串全部在 context | {flat['all_expected_present_rate']:.3f} | {dedup['all_expected_present_rate']:.3f} | {parent['all_expected_present_rate']:.3f} |
| context token 中位数 | {flat['context_tokens_median']:.0f} | {dedup['context_tokens_median']:.0f} | {parent['context_tokens_median']:.0f} |
| context token P95 | {flat['context_tokens_p95']:.0f} | {dedup['context_tokens_p95']:.0f} | {parent['context_tokens_p95']:.0f} |
| 触发 token budget 的题数 | {flat['queries_with_budget_drops']} | {dedup['queries_with_budget_drops']} | {parent['queries_with_budget_drops']} |
| 独立文档数中位数 | {flat['unique_documents_median']:.1f} | {dedup['unique_documents_median']:.1f} | {parent['unique_documents_median']:.1f} |
| 投影延迟中位数 | {flat['projection_latency_median_ms']:.2f} ms | {dedup['projection_latency_median_ms']:.2f} ms | {parent['projection_latency_median_ms']:.2f} ms |

## 直接结论

- parent 让 context token 中位数变化 {delta_tokens:+.0f}。
- 关键答案支持率变化 {delta_support:+.3f}。
- 排除 parent 去重后，扩大正文自身让 Recall@5 变化 {expansion_recall:+.3f}，
  关键答案支持率变化 {expansion_support:+.3f}。
- 检索只执行一次，因此 IR 差异来自 parent 合并和 token 预算，不是两次召回波动。

> “关键答案字符串存在”是确定性支持度代理，不等于完整答案质量；若它有明确收益，
> 再对失败/改善样本跑少量 LLM 端到端评测，而不是一上来对 58 题多调用模型。
"""


def main() -> int:
    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))
    documents = [(path.name, path.read_text(encoding="utf-8"))
                 for path in sorted(CORPUS.glob("*.md"))]
    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"parent_ablation_{tag}", "pw123456", is_admin=False)
    source_name = f"ablation/parent-child/{tag}"

    try:
        started = time.perf_counter()
        child_count = chain.ingest_many_raw(
            documents, source_name=source_name, visibility="private",
            owner_id=str(user.user_id))
        ingest_seconds = time.perf_counter() - started
        print(f"语料 {len(documents)} 篇 / child {child_count} 个，入库 {ingest_seconds:.1f}s")
        print(f"探针 {len(probes)} 题；每题只检索一次，共享 child 排名\n")

        arm_rows = {"flat_child": [], "child_parent_dedup": [], "small_to_big": []}
        retrieval_latencies: list[float] = []
        filters = chain._visibility_filters(user)
        for index, probe in enumerate(probes, start=1):
            started = time.perf_counter()
            nodes = chain._retrieve_nodes(
                user, probe["question"], mode="hybrid", num_queries=1,
                fetch_k=FETCH_K, filters=filters, business={})
            nodes = deduplicate_nodes(nodes)
            retrieval_latencies.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            flat = chain._evidence_from_ranked_nodes(
                user, nodes, TOP_K, expand_parent=False)
            flat_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            dedup = chain._evidence_from_ranked_nodes(
                user, nodes, TOP_K, expand_parent=False, deduplicate_parent=True)
            dedup_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            parent = chain._evidence_from_ranked_nodes(
                user, nodes, TOP_K, expand_parent=True)
            parent_ms = (time.perf_counter() - started) * 1000

            arm_rows["flat_child"].append(_arm_row(probe, flat, flat_ms))
            arm_rows["child_parent_dedup"].append(_arm_row(probe, dedup, dedup_ms))
            arm_rows["small_to_big"].append(_arm_row(probe, parent, parent_ms))
            if index % 10 == 0 or index == len(probes):
                print(f"  {index}/{len(probes)}")

        payload = {
            "corpus_files": len(documents), "child_nodes": child_count,
            "probes": len(probes), "top_k": TOP_K, "fetch_k": FETCH_K,
            "token_budget": get_config().agent_evidence_token_budget,
            "ingest_seconds": round(ingest_seconds, 3),
            "retrieval_latency_median_ms": statistics.median(retrieval_latencies),
            "arms": {name: _aggregate(probes, rows)
                     for name, rows in arm_rows.items()},
            "details": [{"id": probe["id"], "question": probe["question"],
                         "flat_child": arm_rows["flat_child"][index],
                         "child_parent_dedup": arm_rows["child_parent_dedup"][index],
                         "small_to_big": arm_rows["small_to_big"][index]}
                        for index, probe in enumerate(probes)],
        }
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        OUT_MD.write_text(_markdown(payload), encoding="utf-8")
        print("\n" + _markdown(payload))
        print(f"结果：{OUT_JSON.relative_to(ROOT)} / {OUT_MD.relative_to(ROOT)}")
        return 0
    finally:
        chain.delete_source_nodes(owner_id=str(user.user_id), source_name=source_name)
        purge_user(user)
        close_all()


if __name__ == "__main__":
    raise SystemExit(main())
