"""GraphRAG 评测：验证 `eval/GRAPHRAG_HYPOTHESIS.md` 里那四条假设。

**假设书写在动手之前**（git 提交时间早于 `modules/graph_rag/` 的任何代码）。
本脚本只负责填数字，不负责改假设。

## 要填的三行（假设书里的判据）

    1. 全局聚合 3 题：       GraphRAG ___ vs Agentic RAG 0.22
    2. 三跳以上 + 链条到顶 4 题：答对率 ___，LLM 调用 ___ vs Agentic 的 ___
    3. 单点事实 3 题：       GraphRAG ___ vs 向量检索 ___（**预期不赢**）

**三行里只要有一行填不上，这个实验就不算做完。**

★ 第 3 行是设计来防自我欺骗的：如果测出来 GraphRAG 在所有题上都更好，
  那大概率是实验设计有问题（比如两边用了不同语料或不同 embedding），
  而不是它真的全面碾压。

## 受控对比：只换索引结构

| 变量 | 怎么锁的 |
|---|---|
| 语料 | 同一个 `eval/corpus/` |
| embedding | 两条链路都用 `MinibrainEmbedding`（含 MRL 截断 4096→1024） |
| 切分 | 同样的 `chunk_size` / `chunk_overlap` |
| 评测 | 同一批探针、同一套 `ir_metrics` |

## ★ 成本必须一起量，这条最容易被跳过

建图对**每个切片调一次 LLM**，而向量入库是 **0 次**。
`--limit` 先跑小规模看趋势，确认实验设计没问题再跑全量——
本项目因为「没先验证就跑」浪费过一次 13 小时。

跑法：
    uv run --no-sync python scripts/ablate_graphrag.py --limit 20   # 先验证
    uv run --no-sync python scripts/ablate_graphrag.py              # 全量
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

from ir_metrics import summarize                                  # noqa: E402
from minibrain import identity                                    # noqa: E402
from minibrain.db import close_all                                # noqa: E402
from minibrain.modules.graph_rag import chain as graph            # noqa: E402
from minibrain.modules.vector_rag import chain as vector          # noqa: E402
from minibrain.modules.vector_rag import embeddings               # noqa: E402
from minibrain.scripts_purge import purge_user                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"
PROBES = ROOT / "eval" / "probes.json"
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"

TOP_K = 5


def warm(probes) -> None:
    """查询向量预热。★ 评测专用缓存，生产没有。"""
    cache: dict[str, list[float]] = {}
    original = embeddings.embed_query

    def cached(text: str):
        if text not in cache:
            cache[text] = original(text)
        return cache[text]

    embeddings.embed_query = cached
    vector.embed_query = cached
    for p in probes:
        cached(p["question"])
    print(f"  {len(probes)} 个查询向量已预热\n")


def files_of(evidence) -> list[str]:
    seen, out = set(), []
    for e in evidence:
        name = e.location.split(" #")[0]
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def run(search_fn, user, probes, label: str) -> dict:
    pairs, latencies, failures = [], [], 0
    for p in probes:
        started = time.perf_counter()
        try:
            result = search_fn(user, p["question"], top_k=TOP_K)
        except Exception as exc:                                  # noqa: BLE001
            failures += 1
            print(f"      ✗ {p['id']}: {type(exc).__name__}: {exc}")
            continue
        latencies.append((time.perf_counter() - started) * 1000)
        pairs.append((set(p["required"]), files_of(result.evidence)))

    if not pairs:
        return {"failures": failures, "latency_median_ms": 0}
    return {
        "failures": failures,
        "latency_median_ms": statistics.median(latencies),
        **summarize(pairs, (3, 5, 10)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只用前 N 篇语料（先小规模验证实验设计）")
    ap.add_argument("--rebuild", action="store_true",
                    help="强制重建图。默认复用已持久化的图——建图很贵")
    args = ap.parse_args()

    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))

    files = sorted(CORPUS.glob("*.md"))
    if args.limit:
        files = files[:args.limit]
    texts = {f.name: f.read_text(encoding="utf-8") for f in files}

    # ★ 只保留「必需文档全都在本次语料里」的探针。
    #   否则小规模跑的时候，题目要的文档压根没入库，两条链路一起挂零，
    #   比出来的差异毫无意义——**这不是过滤掉难题，是过滤掉无效题**。
    have = set(texts)
    usable = [p for p in probes if set(p["required"]) <= have]
    print(f"语料 {len(texts)} 篇 | 探针 {len(usable)}/{len(probes)} 题可用 | top-{TOP_K}\n")
    if not usable:
        raise SystemExit("没有可用探针——语料太少，必需文档都不在里面")

    user = identity.create_user(f"gr_{uuid.uuid4().hex[:6]}", "pw123456")
    try:
        warm(usable)

        # ---- 向量链路（对照组）----
        vector.drop_all()
        t0 = time.perf_counter()
        for name, text in texts.items():
            vector.ingest(user, name, text)
        vector_ingest_s = time.perf_counter() - t0
        print(f"  向量入库：{vector_ingest_s:.0f}s，LLM 调用 0 次\n")

        # ---- 图链路 ----
        if args.rebuild:
            graph.drop_all()
        try:
            graph._require_index()
            build_stats = {"reused": True}
            print("  复用已持久化的图（--rebuild 可强制重建）\n")
        except Exception:                                         # noqa: BLE001
            print("  建图中…（每个切片调一次 LLM，慢）")
            build_stats = graph.build(texts, show_progress=True)
            print(f"  建图：{build_stats['build_seconds']:.0f}s，"
                  f"LLM 调用约 {build_stats['llm_calls_estimate']} 次，"
                  f"实体 {build_stats['entities']}，三元组 {build_stats['triplets']}\n")

        results = {
            "向量检索": run(vector.search, user, usable, "vector"),
            "图检索": run(graph.search, user, usable, "graph"),
        }

        print("\n" + "=" * 78)
        print("GraphRAG vs 向量检索（同语料、同 embedding、同探针）")
        print("=" * 78)
        print(f"  {'链路':<12}{'CompRec@3':>11}{'CompRec@5':>11}{'Recall@5':>10}"
              f"{'NDCG@5':>9}{'MRR':>8}{'延迟':>11}{'失败':>7}")
        for name, r in results.items():
            if "complete_recall@5" not in r:
                print(f"  {name:<12}{'（全部失败）':>50}")
                continue
            print(f"  {name:<12}{r['complete_recall@3']:>11.3f}{r['complete_recall@5']:>11.3f}"
                  f"{r['recall@5']:>10.3f}{r['ndcg@5']:>9.3f}{r['mrr']:>8.3f}"
                  f"{r['latency_median_ms']:>9.0f}ms{r['failures']:>7}")

        print("\n  ★ 成本对比（假设书里要求必须报的）：")
        print(f"    向量入库：{vector_ingest_s:>7.0f}s   LLM 调用 0 次")
        if "build_seconds" in build_stats:
            print(f"    建图：    {build_stats['build_seconds']:>7.0f}s   "
                  f"LLM 调用约 {build_stats['llm_calls_estimate']} 次")
            ratio = build_stats["build_seconds"] / max(vector_ingest_s, 1e-9)
            print(f"    → 建图比向量入库慢 {ratio:.1f} 倍，且多花 "
                  f"{build_stats['llm_calls_estimate']} 次 LLM 调用")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"ablate_graphrag{'_limit' + str(args.limit) if args.limit else ''}.json"
        out.write_text(json.dumps(
            {"corpus_files": len(texts), "probes_usable": len(usable),
             "probes_total": len(probes), "top_k": TOP_K,
             "vector_ingest_seconds": vector_ingest_s,
             "graph_build": build_stats, "arms": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入 {out.relative_to(ROOT)}")
        return 0
    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
