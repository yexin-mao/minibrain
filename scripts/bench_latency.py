"""Benchmark the current pgvector + sparse-index retrieval path.

The benchmark grows one corpus incrementally, reports cold/warm latency at each
scale, separates retrieval stages from the embedding network call, and includes
a small concurrent load check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")

from minibrain import gateway, identity  # noqa: E402
from minibrain.db import close_all  # noqa: E402
from minibrain.evaluation.provenance import build_provenance  # noqa: E402
from minibrain.modules.vector_rag import chain, core  # noqa: E402
from minibrain.scripts_purge import purge_user  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT = ROOT / "eval" / "results" / "bench_latency.json"
QUERIES = [
    "二线城市住宿一晚最多报多少钱？",
    "MTG-20260617-02 这次会议的决议是什么？",
    "公司一共有几个部门，分别是什么？",
    "张敏的上级的上级是谁？",
    "X7-Pro 是什么产品？当前版本是多少？",
]


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, int(len(ordered) * .95 + .999999) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def maybe_summarize(samples: list[float]) -> dict[str, float] | None:
    return summarize(samples) if samples else None


def _search(user, query: str) -> tuple[float, dict[str, float]]:
    started = time.perf_counter()
    result = gateway.search("vector-rag", user, query, top_k=5, explain=True)
    total = (time.perf_counter() - started) * 1000
    stages = {
        f"{index:02d}:{stage.name}": stage.latency_ms
        for index, stage in enumerate(result.retrieval_trace.stages, start=1)
    } if result.retrieval_trace else {}
    return total, stages


def _attempt_search(user, query: str) -> tuple[float | None, dict[str, float], str | None]:
    try:
        total, stages = _search(user, query)
        return total, stages, None
    except Exception as exc:
        code = getattr(exc, "code", None)
        return None, {}, str(code or type(exc).__name__)


def _measure(user, *, repeat: int, workers: int,
             concurrent_rounds: int) -> dict:
    cold_total, _, cold_error = _attempt_search(user, QUERIES[0])

    totals: list[float] = []
    stage_samples: dict[str, list[float]] = defaultdict(list)
    warm_errors: Counter[str] = Counter()
    for _ in range(repeat):
        for query in QUERIES:
            total, stages, error = _attempt_search(user, query)
            if error is not None:
                warm_errors[error] += 1
                continue
            assert total is not None
            totals.append(total)
            for name, value in stages.items():
                stage_samples[name].append(value)

    concurrent_totals: list[float] = []
    batch_wall: list[float] = []
    batch_successes: list[int] = []
    concurrent_errors: Counter[str] = Counter()
    workload = [QUERIES[index % len(QUERIES)] for index in range(workers)]
    for _ in range(concurrent_rounds):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda query: _attempt_search(user, query), workload))
        wall = (time.perf_counter() - started) * 1000
        batch_wall.append(wall)
        successes = 0
        for total, _stages, error in results:
            if error is not None:
                concurrent_errors[error] += 1
            else:
                assert total is not None
                concurrent_totals.append(total)
                successes += 1
        batch_successes.append(successes)

    warm_attempts = repeat * len(QUERIES)
    concurrent_attempts = workers * concurrent_rounds
    median_wall_seconds = statistics.median(batch_wall) / 1000

    return {
        "cold_query_ms": cold_total,
        "cold_error": cold_error,
        "warm_end_to_end_ms": maybe_summarize(totals),
        "warm_errors": {
            "attempts": warm_attempts,
            "failed": sum(warm_errors.values()),
            "error_rate": sum(warm_errors.values()) / warm_attempts,
            "by_code": dict(warm_errors),
        },
        "stage_ms": {name: summarize(values)
                     for name, values in sorted(stage_samples.items())},
        "concurrency": {
            "workers": workers,
            "rounds": concurrent_rounds,
            "request_latency_ms": maybe_summarize(concurrent_totals),
            "batch_wall_ms": summarize(batch_wall),
            "attempted_queries_per_second": workers / median_wall_seconds,
            "successful_queries_per_second": statistics.median(batch_successes) /
                median_wall_seconds,
            "failed": sum(concurrent_errors.values()),
            "error_rate": sum(concurrent_errors.values()) / concurrent_attempts,
            "errors_by_code": dict(concurrent_errors),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--concurrent-rounds", type=int, default=3)
    args = parser.parse_args()
    if args.repeat <= 0 or args.workers <= 0 or args.concurrent_rounds <= 0:
        parser.error("repeat/workers/concurrent-rounds 必须为正数")

    files = sorted(CORPUS.glob("*.md"))
    scales = sorted(set(min(len(files), value) for value in (30, 84, len(files))))
    user = identity.create_user(
        f"bench_{uuid.uuid4().hex[:6]}", "pw123456", is_admin=False)
    source = core.create_source(user, f"benchmark/latency/{uuid.uuid4().hex[:8]}")
    source_name = str(source["name"])
    rows = []
    ingested = 0
    try:
        for scale in scales:
            batch = files[ingested:scale]
            started = time.perf_counter()
            new_chunks = chain.ingest_many_raw(
                [(path.name, path.read_text(encoding="utf-8")) for path in batch],
                source_name=source_name, visibility="private", owner_id=user.user_id,
            )
            ingest_seconds = time.perf_counter() - started
            ingested = scale
            measurement = _measure(
                user, repeat=args.repeat, workers=args.workers,
                concurrent_rounds=args.concurrent_rounds)
            row = {
                "corpus_files": scale,
                "new_chunks": new_chunks,
                "ingest_seconds": ingest_seconds,
                **measurement,
            }
            rows.append(row)
            print(
                f"{scale:>3} files | warm p50 "
                f"{(measurement['warm_end_to_end_ms'] or {}).get('median', float('nan')):.1f}ms | "
                f"p95 {(measurement['warm_end_to_end_ms'] or {}).get('p95', float('nan')):.1f}ms | "
                f"{args.workers}-way "
                f"{measurement['concurrency']['successful_queries_per_second']:.2f} qps | "
                f"errors {measurement['concurrency']['error_rate']:.1%}",
                flush=True,
            )

            # 每一档完成就保存；后续档即使进程或容器异常，已有证据也不会丢。
            from minibrain.config import get_config

            cfg = get_config()
            partial = {
                "status": "partial",
                "provenance": build_provenance(ROOT, scope="retrieval_latency", config={
                    "embedding_model": cfg.embedding_model,
                    "embedding_dimensions": cfg.embedding_dimensions,
                    "hnsw_ef_search": cfg.hnsw_ef_search,
                    "chunk_size": cfg.chunk_size,
                    "chunk_overlap": cfg.chunk_overlap,
                }),
                "queries": QUERIES,
                "repeat_per_query": args.repeat,
                "rows": rows,
            }
            OUT.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

        from minibrain.config import get_config

        cfg = get_config()
        report = {
            "status": "completed",
            "provenance": build_provenance(ROOT, scope="retrieval_latency", config={
                "embedding_model": cfg.embedding_model,
                "embedding_dimensions": cfg.embedding_dimensions,
                "hnsw_ef_search": cfg.hnsw_ef_search,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
            }),
            "queries": QUERIES,
            "repeat_per_query": args.repeat,
            "rows": rows,
        }
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {OUT.relative_to(ROOT)}")
        return 0
    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
