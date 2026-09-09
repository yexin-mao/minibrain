"""在锁定语料、切分和题集的前提下，公平比较多个 Embedding 模型。

这个脚本刻意绕开 PostgreSQL：每个 arm 都在内存里对同一批 child chunks 做
cosine 检索，避免换模型时数据库维度、HNSW 状态或历史脏数据混进结论。

运行：
  uv run --no-sync python scripts/compare_embeddings.py \
    --config eval/embedding_models.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ir_metrics import summarize  # noqa: E402
from minibrain.config import get_config  # noqa: E402
from minibrain.evaluation.provenance import build_provenance  # noqa: E402
from minibrain.modules.vector_rag.hierarchy import build_hierarchy  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
PROBE_FILES = (ROOT / "eval" / "probes.json", ROOT / "eval" / "probes_blindspot.json")
DEFAULT_OUT = ROOT / "eval" / "results" / "embedding-models.json"
K_GRID = (3, 5, 10)
BATCH_SIZE = 16


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    dimensions: int | None = None
    transform: str = "native_l2"
    price_per_million_tokens: float | None = None


def _l2(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def transform_vector(vector: list[float], arm: Arm) -> list[float]:
    """应用明确声明的变换；绝不为维度对不上静默补零或截断。"""
    if arm.transform not in {"native_l2", "truncate_l2"}:
        raise ValueError(f"{arm.name}: 未知 transform {arm.transform!r}")
    if arm.transform == "native_l2":
        if arm.dimensions is not None and len(vector) != arm.dimensions:
            raise ValueError(
                f"{arm.name}: 期望原生 {arm.dimensions} 维，实际 {len(vector)} 维")
        return _l2(vector)
    if arm.dimensions is None:
        raise ValueError(f"{arm.name}: truncate_l2 必须声明 dimensions")
    if len(vector) < arm.dimensions:
        raise ValueError(
            f"{arm.name}: 只能截断，不能把 {len(vector)} 维补成 {arm.dimensions} 维")
    return _l2(vector[:arm.dimensions])


def _load_arms(path: pathlib.Path) -> list[Arm]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("arms", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("配置至少需要两个 arms")
    arms = [Arm(**row) for row in rows]
    names = [arm.name for arm in arms]
    if len(names) != len(set(names)):
        raise ValueError("arm name 不能重复")
    return arms


def _load_dataset() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    cfg = get_config()
    chunks: list[dict[str, str]] = []
    for path in sorted(CORPUS.glob("*.md")):
        children, _parents = build_hierarchy(
            filename=path.name,
            text=path.read_text(encoding="utf-8"),
            source_name="embedding-model-comparison",
            visibility="private",
            owner_id="offline-eval",
            document_id=None,
            child_size=cfg.chunk_size,
            child_overlap=cfg.chunk_overlap,
            parent_size=cfg.parent_chunk_size,
            document_metadata={},
        )
        chunks.extend({"filename": path.name, "text": node.text} for node in children)

    probes: list[dict[str, Any]] = []
    for path in PROBE_FILES:
        probes.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks, probes


def _embed(client: OpenAI, arm: Arm, texts: list[str]) -> tuple[list[list[float]], dict]:
    vectors: list[list[float]] = []
    latencies: list[float] = []
    prompt_tokens = 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        began = time.perf_counter()
        response = client.embeddings.create(model=arm.model, input=batch)
        latencies.append((time.perf_counter() - began) * 1000)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(transform_vector(item.embedding, arm) for item in ordered)
        usage = getattr(response, "usage", None)
        prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)

    estimated_cost = None
    if arm.price_per_million_tokens is not None and prompt_tokens:
        estimated_cost = prompt_tokens / 1_000_000 * arm.price_per_million_tokens
    return vectors, {
        "requests": len(latencies),
        "latency_ms": {
            "total": sum(latencies),
            "median_per_batch": statistics.median(latencies),
            "p95_per_batch": sorted(latencies)[max(0, math.ceil(len(latencies) * .95) - 1)],
        },
        "prompt_tokens": prompt_tokens or None,
        "estimated_cost_usd": estimated_cost,
    }


def rank_documents(query: list[float], chunk_vectors: list[list[float]],
                   chunks: list[dict[str, str]]) -> list[str]:
    scored = sorted(
        ((sum(left * right for left, right in zip(query, vector)), index)
         for index, vector in enumerate(chunk_vectors)),
        reverse=True,
    )
    seen: set[str] = set()
    ranking: list[str] = []
    for _score, index in scored:
        filename = chunks[index]["filename"]
        if filename not in seen:
            seen.add(filename)
            ranking.append(filename)
    return ranking


def evaluate_arm(client: OpenAI, arm: Arm, chunks: list[dict[str, str]],
                 probes: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    chunk_vectors, corpus_usage = _embed(client, arm, [row["text"] for row in chunks])
    query_vectors, query_usage = _embed(client, arm, [row["question"] for row in probes])
    rankings = {
        probe["id"]: rank_documents(query, chunk_vectors, chunks)
        for probe, query in zip(probes, query_vectors)
    }
    pairs = [(set(row["required"]), rankings[row["id"]]) for row in probes]
    return {
        "name": arm.name,
        "model": arm.model,
        "dimensions": len(chunk_vectors[0]),
        "transform": arm.transform,
        "metrics": summarize(pairs, K_GRID),
        "corpus_embedding": corpus_usage,
        "query_embedding": query_usage,
        "wall_seconds": time.perf_counter() - started,
        "rankings": rankings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-probes", type=int, help="只用于冒烟，正式报告不要传")
    args = parser.parse_args()

    arms = _load_arms(args.config)
    chunks, probes = _load_dataset()
    if args.limit_probes:
        probes = probes[:args.limit_probes]
    cfg = get_config()
    if not cfg.embedding_configured:
        parser.error("EMBEDDING_API_KEY 未配置")
    client = OpenAI(
        base_url=cfg.embedding_base_url,
        api_key=cfg.embedding_api_key,
        timeout=cfg.embedding_timeout_seconds,
        max_retries=cfg.llm_max_retries,
    )

    print(f"语料 {len(chunks)} chunks / {len(set(r['filename'] for r in chunks))} files"
          f" | 探针 {len(probes)} 题 | arms {len(arms)}", flush=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def save(status: str) -> None:
        report = {
            "schema_version": "embedding-model-comparison-v1",
            "status": status,
            "provenance": build_provenance(ROOT, scope="embedding_model_comparison", config={
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "parent_chunk_size": cfg.parent_chunk_size,
                "probe_files": [str(path.relative_to(ROOT)) for path in PROBE_FILES],
            }),
            "method": {
                "retrieval": "dense cosine over identical child chunks; document-level dedup",
                "query_prefix": None,
                "limitations": [
                    "这是当前生产输入口径的 drop-in 对比，没有为各模型添加专属 query instruction",
                    "API 延迟包含供应商排队和网络波动；价格仅在配置显式提供时估算",
                ],
            },
            "corpus_chunks": len(chunks),
            "corpus_files": len(set(row["filename"] for row in chunks)),
            "probes": len(probes),
            "arms": rows,
            "failures": failures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for arm in arms:
        try:
            row = evaluate_arm(client, arm, chunks, probes)
        except Exception as exc:
            failure = {
                "name": arm.name,
                "model": arm.model,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            failures.append(failure)
            save("partial")
            print(f"{arm.name}: FAILED {failure['error_type']}: {failure['error']}",
                  file=sys.stderr, flush=True)
            continue
        rows.append(row)
        metrics = row["metrics"]
        print(f"{arm.name}: MRR={metrics['mrr']:.3f} "
              f"Recall@5={metrics['recall@5']:.3f} "
              f"CompleteRecall@5={metrics['complete_recall@5']:.3f} "
              f"NDCG@5={metrics['ndcg@5']:.3f}", flush=True)
        save("partial")  # 每个 arm 完成即落盘，后续供应商失败也不丢已完成结果。

    final_status = "smoke" if args.limit_probes else "completed"
    if failures:
        final_status = "partial"
    save(final_status)
    print(f"结果已写入 {args.output}")
    if len(rows) < 2:
        print("至少需要两个成功的 arm 才能形成对比", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
