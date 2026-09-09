"""在官方 NanoNQ / NanoHotpotQA 上评测 Minibrain 的文档检索。

默认只比较检索部件，不调用生成式 LLM，也不使用业务 metadata：

  uv run --extra public-eval python scripts/eval_nanobeir.py --download-only
  uv run --extra public-eval python scripts/eval_nanobeir.py
  uv run --extra public-eval --extra rerank-ce python scripts/eval_nanobeir.py --with-ce

指标严格按原始 corpus document 计算。一个文档被切成多个 chunk 后，只保留其
最高排名 chunk；否则 chunk 重复会虚增或歪曲公开榜单的 document-level 指标。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import uuid

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ir_metrics import summarize  # noqa: E402
from nanobeir_data import (  # noqa: E402
    SPECS, NanoBeirDataset, aggregate_document_ranking,
    document_filename, download_and_load,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "eval" / "cache" / "nanobeir"
DEFAULT_OUTPUT = ROOT / "eval" / "results" / "nanobeir.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--with-ce", action="store_true",
                        help="增加 hybrid+CE 和交互组；需安装 rerank-ce extra")
    parser.add_argument("--limit-queries", type=int, default=0,
                        help="仅供管线 smoke；非标准评测，结果会显式标记")
    return parser.parse_args()


def dataset_summary(dataset: NanoBeirDataset) -> dict:
    relevant_counts = [len(v) for v in dataset.qrels.values()]
    return {
        "repo_id": dataset.spec.repo_id,
        "revision": dataset.spec.revision,
        "documents": len(dataset.corpus),
        "queries": len(dataset.queries),
        "relevant_documents_mean": sum(relevant_counts) / len(relevant_counts),
        "relevant_documents_min": min(relevant_counts),
        "relevant_documents_max": max(relevant_counts),
    }


def _arms(with_ce: bool) -> list[tuple[str, dict]]:
    # 所有组使用相同的 40-document 候选池；CE-only 才是 cross-encoder 的主对照。
    arms = [
        ("vector", {"mode": "vector", "use_mmr": False}),
        ("hybrid", {"mode": "hybrid", "use_mmr": False}),
        ("hybrid+mmr", {"mode": "hybrid", "use_mmr": True}),
    ]
    if with_ce:
        arms.append(("hybrid+ce", {
            "mode": "hybrid", "use_mmr": False,
            "rerank_pool": 40, "reranker": "ce",
        }))
    return arms


def _warm_query_embeddings(queries: list[str]):
    """各 arm 共用查询向量，使消融延迟不被网络波动和执行顺序污染。"""
    from minibrain import llamaindex_setup

    original = llamaindex_setup.embed_query
    cache: dict[str, list[float]] = {}

    def cached(text: str) -> list[float]:
        if text not in cache:
            cache[text] = original(text)
        return cache[text]

    llamaindex_setup.embed_query = cached
    for query in queries:
        cached(query)
    return original


def _run_arm(user, dataset: NanoBeirDataset, query_ids: list[str],
             filename_to_doc_id: dict[str, str], kwargs: dict) -> dict:
    from minibrain.modules.vector_rag import core

    pairs: list[tuple[set[str], list[str]]] = []
    latencies: list[float] = []
    rankings: dict[str, list[str]] = {}
    for query_id in query_ids:
        started = time.perf_counter()
        result = core.search(
            user, dataset.queries[query_id], top_k=10,
            candidate_pool=40, metadata_filtering=False, **kwargs)
        latencies.append((time.perf_counter() - started) * 1000)
        ranking = aggregate_document_ranking(
            (e.location for e in result.evidence), filename_to_doc_id)
        rankings[query_id] = ranking
        pairs.append((set(dataset.qrels[query_id]), ranking))
    return {
        "metrics": summarize(pairs, (3, 5, 10)),
        "latency_median_ms_without_query_embedding": statistics.median(latencies),
        "latency_p95_ms_without_query_embedding": sorted(latencies)[
            max(0, int(len(latencies) * 0.95 + 0.999999) - 1)],
        "rankings": rankings,
    }


def run_task(dataset: NanoBeirDataset, *, with_ce: bool,
             limit_queries: int) -> dict:
    from minibrain import identity, llamaindex_setup
    from minibrain.modules.vector_rag import chain, core
    from minibrain.scripts_purge import purge_user

    tag = uuid.uuid4().hex[:8]
    user = identity.create_user(f"nanobeir_{tag}", "pw123456", is_admin=False)
    original_embed_query = None
    try:
        source = core.create_source(user, f"benchmark/{dataset.spec.name}/{tag}")
        filename_to_doc_id = {
            document_filename(dataset.spec.name, doc.doc_id): doc.doc_id
            for doc in dataset.corpus
        }
        documents = [
            (document_filename(dataset.spec.name, doc.doc_id), doc.content)
            for doc in dataset.corpus
        ]
        started = time.perf_counter()
        chunks = chain.ingest_many_raw(
            documents, source_name=str(source["name"]), visibility="private",
            owner_id=user.user_id)
        ingest_seconds = time.perf_counter() - started

        query_ids = list(dataset.queries)
        if limit_queries > 0:
            query_ids = query_ids[:limit_queries]
        original_embed_query = _warm_query_embeddings(
            [dataset.queries[query_id] for query_id in query_ids])

        results = {}
        for label, kwargs in _arms(with_ce):
            print(f"  {dataset.spec.name}: {label}", flush=True)
            results[label] = _run_arm(
                user, dataset, query_ids, filename_to_doc_id, kwargs)
        return {
            **dataset_summary(dataset),
            "standard_run": limit_queries == 0,
            "evaluated_queries": len(query_ids),
            "chunks": chunks,
            "ingest_seconds": ingest_seconds,
            "candidate_pool": 40,
            "top_k": 10,
            "metadata_filtering": False,
            "query_embedding_excluded_from_latency": True,
            "arms": results,
        }
    finally:
        if original_embed_query is not None:
            llamaindex_setup.embed_query = original_embed_query
        purge_user(user)


def main() -> int:
    args = parse_args()
    datasets = []
    for task in args.tasks:
        print(f"下载/校验 {task} ({SPECS[task].revision[:12]}) ...", flush=True)
        dataset = download_and_load(SPECS[task], args.cache_dir)
        datasets.append(dataset)
        summary = dataset_summary(dataset)
        print(f"  {summary['documents']} documents | {summary['queries']} queries | "
              f"relevant/query {summary['relevant_documents_mean']:.2f}")

    if args.download_only:
        print("数据下载与 schema/qrels 校验通过；未调用 embedding 或 LLM。")
        return 0

    from minibrain.config import get_config
    from minibrain.evaluation.provenance import build_provenance

    cfg = get_config()
    output = {
        "benchmark": "selected NanoBEIR tasks (not the full NanoBEIR aggregate)",
        "provenance": build_provenance(ROOT, scope="retrieval", config={
            "embedding_model": cfg.embedding_model,
            "embedding_dimensions": cfg.embedding_dimensions,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "parent_chunk_size": cfg.parent_chunk_size,
            "expand_parent_context": cfg.expand_parent_context,
            "hnsw_ef_search": cfg.hnsw_ef_search,
        }),
        "tasks": {},
    }
    for dataset in datasets:
        output["tasks"][dataset.spec.name] = run_task(
            dataset, with_ce=args.with_ce, limit_queries=args.limit_queries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        try:
            from minibrain.db import close_all
        except ImportError:
            pass
        else:
            close_all()
