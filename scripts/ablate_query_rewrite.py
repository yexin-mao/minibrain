"""查询改写消融：改写查询到底值不值，以及它必须打赢谁。

## 为什么做

RAG 是一条链，本项目已经量过其中四段，唯独**查询侧一次没碰过**：

    查询 → [空白] → 召回 → 融合 → 重排 → 生成
            ↑        探针一  探针一之四  初步   探针六

而「RAG 检索效果不好怎么优化」是面试必问题，完整答案有三个方向：
改查询、改召回、改排序。只答后两个是不完整的。

## 更重要的：已有数据指出它该做

轨迹评测（34 用例）测出来：

    第 1 轮：2.31 条新证据
    第 2 轮：2.15 条新证据      ← 几乎和第一轮一样多

**第二轮之所以能捞到这么多新东西，是因为 agent 换了个说法重新查。**
那本质上就是查询改写——只不过是用「多一次 LLM 调用 + 一整轮完整检索」
换来的。如果在第一轮就把改写做掉，可能一轮就够。

★ 所以本消融里**必须有 agentic 双轮这一组**。和 rerank 消融里放
  baseline@10 是同一个道理：**新方法要证明的不是「比什么都不做好」，
  是「比现有的便宜方案好」**。

## 四组对照

| 组 | 配置 | 回答什么 |
|---|---|---|
| baseline | num_queries=1 | 现状 |
| multi-query | num_queries=3 | LLM 生成 2 条改写 + 原问题，各查一遍再 RRF |
| HyDE | 先编假答案再检索 | 术语丰富的假答案能不能捞得更准 |
| agentic 双轮 | 现在的 agent | **改写单轮能不能打赢它** |

## 三个已经踩过的坑（都在 chain.py 里修了）

1. **框架默认的改写提示词是英文的**。中文语料下拿它改写，模型很可能吐英文
   查询，然后拿英文检索中文文档，召回直接塌。所以自己写了中文版。
2. **LlamaIndex 的 `OpenAI` 类硬编码了 OpenAI 官方模型名单**，
   `deepseek/deepseek-v4-flash` 不在里面，一开改写就 ValueError。
   换 `OpenAILike`。这个坑藏得深：num_queries=1 时碰不到。
3. 查询 embedding 要预热缓存，否则各组之间的延迟差异大半是 API 抖动。

跑法：uv run --no-sync python scripts/ablate_query_rewrite.py
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
from minibrain.modules.vector_rag import chain, embeddings         # noqa: E402
from minibrain.scripts_purge import purge_user                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"
PROBES = ROOT / "eval" / "probes.json"
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"

TOP_K = 5

# (标签, num_queries, hyde, 每题额外的 LLM 调用次数)
ARMS = [
    ("baseline",        1, False, 0),
    ("multi-query ×3",  3, False, 1),   # 一次调用生成 2 条改写
    ("HyDE",            1, True,  1),   # 一次调用编假答案
    ("HyDE + ×3",       3, True,  2),   # 两者叠加，看会不会互相干扰
]


def warm_query_embeddings(probes) -> None:
    """查询向量预热。★ 评测专用缓存，生产没有。

    不预热的话各组之间的延迟差异大半是 embedding API 的抖动
    （实测中位数 1.2s，p95 4.9s），而不是配置差异。
    """
    cache: dict[str, list[float]] = {}
    original = embeddings.embed_query

    def cached(text: str):
        if text not in cache:
            cache[text] = original(text)
        return cache[text]

    embeddings.embed_query = cached
    chain.embed_query = cached

    started = time.perf_counter()
    for p in probes:
        cached(p["question"])
    print(f"  {len(probes)} 个查询向量已预热，用时 {time.perf_counter() - started:.0f}s")
    print("  ★ 下面的延迟不含原问题的 embedding；改写产生的新查询仍要实时算\n")


def ingest(user) -> int:
    files = sorted(CORPUS.glob("*.md"))
    for path in files:
        chain.ingest(user, path.name, path.read_text(encoding="utf-8"))
    return len(files)


def ceilings(probes, ks) -> dict:
    """理论上限。不先算它，就会拿一个够不着的目标衡量改动。"""
    n = len(probes)
    return {k: {
        "complete_recall": sum(1 for p in probes if len(p["required"]) <= k) / n,
        "recall": sum(min(k, len(p["required"])) / len(p["required"]) for p in probes) / n,
    } for k in ks}


def files_of(evidence) -> list[str]:
    seen, out = set(), []
    for e in evidence:
        name = e.location.split(" #")[0]
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def run_arm(user, probes, num_queries: int, hyde: bool) -> dict:
    pairs, latencies, chars, failures = [], [], [], 0
    for p in probes:
        started = time.perf_counter()
        try:
            result = chain.search(user, p["question"], top_k=TOP_K,
                                  num_queries=num_queries, hyde=hyde)
        except Exception as exc:                                  # noqa: BLE001
            # ★ 改写多了一次 LLM 调用，就多了一个失败点。失败要计数不要吞掉——
            #   一组「指标很好但三分之一的题报错了」不是好结果。
            failures += 1
            print(f"      ✗ {p['id']}: {type(exc).__name__}")
            continue
        latencies.append((time.perf_counter() - started) * 1000)
        pairs.append((set(p["required"]), files_of(result.evidence)))
        chars.append(sum(len(e.snippet) for e in result.evidence))

    return {
        "failures": failures,
        "latency_median_ms": statistics.median(latencies) if latencies else 0,
        "context_chars_median": statistics.median(chars) if chars else 0,
        **summarize(pairs, (3, 5, 10)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))
    if args.limit:
        probes = probes[:args.limit]

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"qrw_{tag}", "pw123456", is_admin=False)

    try:
        chain.drop_all()
        total = ingest(user)
        print(f"语料 {total} 篇 | 探针 {len(probes)} 题 | top-{TOP_K}\n")
        warm_query_embeddings(probes)

        caps = ceilings(probes, (3, 5, 10))
        print("  理论上限（题目本身决定，改检索改不动）：")
        for k in (3, 5, 10):
            print(f"    k={k:<3} CompRecall 上限 {caps[k]['complete_recall']:.3f}"
                  f"   Recall 上限 {caps[k]['recall']:.3f}")
        print()

        results = {}
        for label, nq, hyde, extra_calls in ARMS:
            print(f"  跑 {label} …")
            results[label] = {**run_arm(user, probes, nq, hyde),
                              "extra_llm_calls_per_query": extra_calls}

        print("\n" + "=" * 94)
        print("查询改写消融：每一列都要和 baseline 比成本，不只比指标")
        print("=" * 94)
        print(f"  {'配置':<16}{'CompRec@3':>11}{'CompRec@5':>11}{'Recall@5':>10}"
              f"{'NDCG@5':>9}{'MRR':>8}{'额外LLM':>9}{'延迟':>11}{'失败':>7}")
        for label, _nq, _h, _e in ARMS:
            r = results[label]
            print(f"  {label:<16}{r['complete_recall@3']:>11.3f}{r['complete_recall@5']:>11.3f}"
                  f"{r['recall@5']:>10.3f}{r['ndcg@5']:>9.3f}{r['mrr']:>8.3f}"
                  f"{r['extra_llm_calls_per_query']:>9}"
                  f"{r['latency_median_ms']:>9.0f}ms{r['failures']:>7}")

        base = results["baseline"]
        print(f"\n  ★ 判据：改写要在 CompRecall@5 上超过 baseline（{base['complete_recall@5']:.3f}），")
        print("    并且涨幅要对得起多花的 LLM 调用。只涨一两个点不值一次额外调用。")
        print("  ★ 「失败」这一列不能忽略：多一次 LLM 调用就多一个失败点。")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "ablate_query_rewrite.json"
        out.write_text(json.dumps(
            {"corpus_files": total, "probes": len(probes), "top_k": TOP_K,
             "ceilings": {str(k): v for k, v in caps.items()}, "arms": results},
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
