"""重排消融：重排到底值不值，以及它必须打赢谁。

## 要回答的问题不是「重排有没有用」

加 reranker 是 RAG 的标配动作，绝大多数项目加了就写进简历。
但真正该问的是：

    **「重排后的 top-5」能不能打赢「不重排的 top-10」？**

因为不加重排、直接把 k 从 3 提到 10，Complete Recall 就从 0.655 涨到 0.948——
**几乎不花钱**。重排要证明的不是"比 top-3 好"（那太容易了），
而是"用更少的片段达到同样甚至更好的效果"。赢在哪：
更少片段 → 更省 token、噪声更少、幻觉更少。

所以对照组里**必须**有 baseline@10。少了它，整个实验没有说服力。

## 天花板：先算清楚空间有多大，再谈涨了多少

58 题里有 12 题需要 ≥4 篇文档（最多一题 11 篇）。
这些题在 k=3 时**数学上不可能**拿满 Complete Recall。所以：

    k=3    上限 0.793   实测 0.655   真实空间 0.138
    k=5    上限 0.931   实测 0.724   真实空间 0.207   ← 空间最大
    k=10   上限 0.983   实测 0.948   真实空间 0.035

★ 本脚本会把上限一起打出来。不算天花板就会把 0.655→0.948
  当成 29 个点的空间去追，那是虚的——**其中一半根本追不到**。

## 位置偏见

LLM 天然偏向排在前面的选项。如果按召回顺序直接喂给它，
它很可能只是"确认"原有排序，那样测出来的收益是假的。
所以生产路径打乱输入顺序（见 rerank.py），而本脚本额外跑一组
**不打乱**的对照，把这个偏见的大小量出来。

跑法：uv run --no-sync python scripts/ablate_rerank.py
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
from minibrain import gateway, identity                           # noqa: E402
from minibrain.db import close_all                                # noqa: E402
from minibrain.modules.vector_rag import core, embeddings         # noqa: E402
from minibrain.handwritten.rerank import rerank            # noqa: E402
from minibrain.scripts_purge import purge_user                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"
PROBES = ROOT / "eval" / "probes.json"
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"

# (标签, top_k, rerank_pool)。rerank_pool=0 表示不重排。
ARMS = [
    ("baseline@3",        3,  0),
    ("baseline@5",        5,  0),
    ("baseline@10",      10,  0),      # ★ 重排必须打赢它，否则不值得做
    ("rerank 20→3",       3, 20),
    ("rerank 20→5",       5, 20),
    ("rerank 10→5",       5, 10),      # 小池子：便宜，但能救回来的少
]


def warm_query_embeddings(probes) -> None:
    """把 58 个问题的向量**一次性算好并缓存**，之后所有配置复用。

    ★★ 这件事有两个理由，第二个才是关键：

    1. 省时间。58 题 × 6 组配置 = 348 次 embedding 调用，而问题从头到尾没变过，
       算 348 次和算 58 次结果完全一样。第一版没做这个，光 baseline@3 一组
       就跑了 14 分钟（embedding 实测中位数 1.2s，但 p95 是 4.9s、最大 16.9s，
       波动极大），六组下来要两个多小时。

    2. **让延迟这一列变得可比。** 这才是关键。
       如果不预热，第一组要付 embedding 的钱、后面几组吃缓存，
       测出来的「延迟」差异大部分是缓存命中率，不是配置差异——
       和 probe_hnsw.py 里踩过的「顺序本身是一个变量」是同一类错误。
       预热之后所有配置一律命中缓存，延迟这一列反映的才是
       **检索 + 重排本身的成本**，也正是这个消融要比的东西。

    ★ 这是评测专用的缓存。生产代码里没有 embedding 缓存（那是待办事项），
      所以下面报出来的延迟**不含** embedding 网络往返，看的时候要记得。
    """
    cache: dict[str, list[float]] = {}
    original = embeddings.embed_query

    def cached(text: str):
        if text not in cache:
            cache[text] = original(text)
        return cache[text]

    embeddings.embed_query = cached
    core.embed_query = cached          # core 是 from ... import 进来的，要单独替换

    started = time.perf_counter()
    for p in probes:
        cached(p["question"])
    print(f"  {len(probes)} 个查询向量已预热，用时 {time.perf_counter() - started:.0f}s")
    print("  ★ 下面的延迟不含 embedding 网络往返（评测专用缓存，生产没有）\n")


def ingest(user) -> int:
    files = sorted(CORPUS.glob("*.md"))
    for path in files:
        gateway.process("vector-rag", gateway.call(
            "vector-rag", "upload_document", user, None,
            path.name, path.read_text(encoding="utf-8")))
    bad = [d for d in gateway.call("vector-rag", "list_documents", user)
           if d["status"] != "ready"]
    if bad:
        raise SystemExit(f"入库失败：{[(d['filename'], d['error']) for d in bad]}")
    return len(files)


def ceilings(probes: list[dict], ks: tuple[int, ...]) -> dict:
    """算每个指标在这套题目下的理论上限。

    ★ 这是本脚本最容易被跳过、但最不该跳过的一步。
      Complete Recall@k 的上限 = 必需文档数 ≤ k 的题目占比；
      Recall@k 的上限 = 每题 min(k, 必需数)/必需数 的平均。
      不先算它，就会拿一个够不着的目标去衡量改动。
    """
    n = len(probes)
    out = {}
    for k in ks:
        out[k] = {
            "complete_recall": sum(1 for p in probes if len(p["required"]) <= k) / n,
            "recall": sum(min(k, len(p["required"])) / len(p["required"])
                          for p in probes) / n,
        }
    return out


def run_arm(user, probes, top_k: int, pool: int) -> dict:
    """跑一组配置，返回指标 + 成本。"""
    pairs, latencies, chars = [], [], []
    for p in probes:
        started = time.perf_counter()
        result = core.search(user, p["question"], top_k=top_k, rerank_pool=pool)
        latencies.append((time.perf_counter() - started) * 1000)

        seen, files = set(), []
        for e in result.evidence:
            name = e.location.split(" #")[0]
            if name not in seen:
                seen.add(name)
                files.append(name)
        pairs.append((set(p["required"]), files))
        # ★ 报「送进 context 的字符数」而不是片段数：重排的卖点是"更少的量
        #   达到同样的效果"，而片段长度不一，只数片段会低估这个差异。
        chars.append(sum(len(e.snippet) for e in result.evidence))

    metrics = summarize(pairs, (3, 5, 10))
    return {
        "latency_median_ms": statistics.median(latencies),
        "context_chars_median": statistics.median(chars),
        **metrics,
    }


def measure_position_bias(user, probes, sample: int) -> dict:
    """量位置偏见：同一批候选，打乱 vs 不打乱送进模型，排序差多少。

    差异大 = 模型确实受展示顺序影响 → 生产路径必须打乱。
    差异小 = 这个模型对位置不敏感，打乱是白做（但也不亏）。
    """
    same_top1 = shuffled_fallbacks = plain_fallbacks = 0
    for p in probes[:sample]:
        result = core.search(user, p["question"], top_k=20)
        snippets = [e.snippet for e in result.evidence]
        if len(snippets) < 2:
            continue
        a = rerank(p["question"], snippets, shuffle=True)
        b = rerank(p["question"], snippets, shuffle=False)
        shuffled_fallbacks += a.fell_back
        plain_fallbacks += b.fell_back
        if a.order and b.order and a.order[0] == b.order[0]:
            same_top1 += 1
    return {"sample": sample, "same_top1": same_top1,
            "shuffled_fallbacks": shuffled_fallbacks,
            "plain_fallbacks": plain_fallbacks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias-sample", type=int, default=10,
                    help="量位置偏见用多少道题（每题 2 次 LLM 调用）")
    args = ap.parse_args()

    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"rerank_{tag}", "pw123456", is_admin=False)

    try:
        total = ingest(user)
        print(f"语料 {total} 篇 | 探针 {len(probes)} 题\n")
        warm_query_embeddings(probes)

        caps = ceilings(probes, (3, 5, 10))
        print("  理论上限（这套题目决定的，改检索改不动）：")
        print(f"  {'k':>4}{'Complete Recall 上限':>22}{'Recall 上限':>16}")
        for k in (3, 5, 10):
            print(f"  {k:>4}{caps[k]['complete_recall']:>22.3f}{caps[k]['recall']:>16.3f}")
        n_hard = sum(1 for p in probes if len(p["required"]) > 3)
        print(f"  （{n_hard} 题需要 >3 篇文档，它们在 k=3 时不可能拿满）\n")

        results = {}
        for label, top_k, pool in ARMS:
            results[label] = run_arm(user, probes, top_k, pool)
            print(f"  跑完 {label}")

        print("\n" + "=" * 92)
        print("重排消融：每一列都要和「baseline@10」比，不是和「baseline@3」比")
        print("=" * 92)
        print(f"  {'配置':<16}{'CompRecall@3':>14}{'CompRecall@5':>14}"
              f"{'Recall@5':>11}{'NDCG@5':>10}{'MRR':>9}"
              f"{'context字符':>13}{'延迟':>11}")
        for label, top_k, pool in ARMS:
            r = results[label]
            print(f"  {label:<16}{r['complete_recall@3']:>14.3f}"
                  f"{r['complete_recall@5']:>14.3f}{r['recall@5']:>11.3f}"
                  f"{r['ndcg@5']:>10.3f}{r['mrr']:>9.3f}"
                  f"{r['context_chars_median']:>13,.0f}{r['latency_median_ms']:>9.0f}ms")

        base = results["baseline@10"]
        print(f"\n  ★ 判据：rerank 20→5 要在 CompRecall@5 上追平或超过 baseline@10"
              f"（{base['complete_recall@5']:.3f}），")
        print("    同时 context 字符数显著更少。只赢一半都不算赢。")

        print("\n  位置偏见（同一批候选，打乱 vs 不打乱）：")
        bias = measure_position_bias(user, probes, args.bias_sample)
        print(f"    {args.bias_sample} 题中，两种顺序选出同一个第一名的：{bias['same_top1']} 题")
        print(f"    解析失败回落次数：打乱 {bias['shuffled_fallbacks']} / "
              f"不打乱 {bias['plain_fallbacks']}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "ablate_rerank.json"
        out.write_text(json.dumps(
            {"corpus_files": total, "probes": len(probes),
             "ceilings": {str(k): v for k, v in caps.items()},
             "arms": results, "position_bias": bias},
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
