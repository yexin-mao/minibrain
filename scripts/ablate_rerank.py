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

跑法：uv run --no-sync python scripts/ablate_rerank.py
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

from ir_metrics import summarize                                  # noqa: E402
from minibrain import identity                                    # noqa: E402
from minibrain.contracts import ModuleError                       # noqa: E402
from minibrain.db import close_all                                # noqa: E402
from minibrain.modules.vector_rag import core                     # noqa: E402
from minibrain.scripts_purge import purge_user                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"
PROBES = ROOT / "eval" / "probes.json"
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"

# 只比较 cross-encoder 这一个变量。所有组固定 40 条候选池、关闭 MMR 和 metadata。
ARMS = [("baseline", 0), ("ce", 40)]


def warm_query_embeddings(probes):
    """把 58 个问题的向量**一次性算好并缓存**，之后所有配置复用。

    ★★ 这件事有两个理由，第二个才是关键：

    1. 省时间。58 题 × 6 组配置 = 348 次 embedding 调用，而问题从头到尾没变过，
       算 232 次和算 58 次结果完全一样。第一版没做这个，光一组 baseline
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
    from minibrain import llamaindex_setup

    original = llamaindex_setup.embed_query

    def cached(text: str):
        if text not in cache:
            cache[text] = original(text)
        return cache[text]

    # 产品检索走 MinibrainEmbedding，它引用的是 llamaindex_setup 模块内导入的函数。
    # 旧脚本只 patch embeddings/core，迁移到 LlamaIndex 后已经不生效，导致每个 arm
    # 仍在调用远程 embedding。这里钉住真正的运行时引用。
    llamaindex_setup.embed_query = cached

    started = time.perf_counter()
    for p in probes:
        cached(p["question"])
    print(f"  {len(probes)} 个查询向量已预热，用时 {time.perf_counter() - started:.0f}s")
    print("  ★ 下面的延迟不含 embedding 网络往返（评测专用缓存，生产没有）\n")
    return original


def ingest(user) -> int:
    from minibrain.modules.vector_rag import chain

    files = sorted(CORPUS.glob("*.md"))
    source = core.create_source(user, f"benchmark/rerank/{uuid.uuid4().hex[:8]}")
    chain.ingest_many_raw(
        [(path.name, path.read_text(encoding="utf-8")) for path in files],
        source_name=str(source["name"]), visibility="private", owner_id=user.user_id,
    )
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


def run_arm(user, probes, pool: int) -> dict:
    """跑一组配置，返回指标 + 成本。

    pool > 0 表示启用可选 cross-encoder；MMR 在所有组都关闭。
    """
    kwargs = {
        "rerank_pool": pool,
        "reranker": "ce",
        "candidate_pool": 40,
        "use_mmr": False,
        # 保持这份历史消融只比较候选选择，不把 metadata 同时混进变量。
        "metadata_filtering": False,
    }

    pairs, latencies, chars5, chars10, rankings = [], [], [], [], []
    for p in probes:
        started = time.perf_counter()
        # 一次取 top-10；@5 是同一排名的前缀，不能为两个 cutoff 重复跑 CE。
        result = core.search(user, p["question"], top_k=10, **kwargs)
        latencies.append((time.perf_counter() - started) * 1000)

        seen, files = set(), []
        for e in result.evidence:
            name = e.location.split(" #")[0]
            if name not in seen:
                seen.add(name)
                files.append(name)
        pairs.append((set(p["required"]), files))
        rankings.append({"question": p["question"], "required": p["required"],
                         "ranking": files})
        # ★ 报「送进 context 的字符数」而不是片段数：重排的卖点是"更少的量
        #   达到同样的效果"，而片段长度不一，只数片段会低估这个差异。
        chars5.append(sum(len(e.snippet) for e in result.evidence[:5]))
        chars10.append(sum(len(e.snippet) for e in result.evidence[:10]))

    metrics = summarize(pairs, (3, 5, 10))
    return {
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[
            max(0, int(len(latencies) * 0.95 + 0.999999) - 1)],
        "context_chars_median@5": statistics.median(chars5),
        "context_chars_median@10": statistics.median(chars10),
        "rankings": rankings,
        **metrics,
    }


def main() -> int:
    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"rerank_{tag}", "pw123456", is_admin=False)
    original_embed_query = None

    try:
        total = ingest(user)
        print(f"语料 {total} 篇 | 探针 {len(probes)} 题\n")
        original_embed_query = warm_query_embeddings(probes)

        caps = ceilings(probes, (3, 5, 10))
        print("  理论上限（这套题目决定的，改检索改不动）：")
        print(f"  {'k':>4}{'Complete Recall 上限':>22}{'Recall 上限':>16}")
        for k in (3, 5, 10):
            print(f"  {k:>4}{caps[k]['complete_recall']:>22.3f}{caps[k]['recall']:>16.3f}")
        n_hard = sum(1 for p in probes if len(p["required"]) > 3)
        print(f"  （{n_hard} 题需要 >3 篇文档，它们在 k=3 时不可能拿满）\n")

        results = {}
        for label, pool in ARMS:
            try:
                results[label] = run_arm(user, probes, pool)
            except ModuleError as exc:
                # 缺 CE 可选依赖时跳过该组，不中断 baseline 实验。
                print(f"  跳过 {label}：{exc.message}")
                continue
            print(f"  跑完 {label}")

        print("\n" + "=" * 92)
        print("cross-encoder 消融：候选池固定 40，MMR/metadata 全部关闭")
        print("=" * 92)
        print(f"  {'配置':<16}{'CompRecall@3':>14}{'CompRecall@5':>14}"
              f"{'Recall@5':>11}{'NDCG@5':>10}{'MRR':>9}"
              f"{'context@5':>12}{'context@10':>12}{'延迟':>11}")
        for label, pool in ARMS:
            r = results.get(label)
            if r is None:
                print(f"  {label:<16}{'（未跑，缺依赖）':>14}")
                continue
            print(f"  {label:<16}{r['complete_recall@3']:>14.3f}"
                  f"{r['complete_recall@5']:>14.3f}{r['recall@5']:>11.3f}"
                  f"{r['ndcg@5']:>10.3f}{r['mrr']:>9.3f}"
                  f"{r['context_chars_median@5']:>12,.0f}"
                  f"{r['context_chars_median@10']:>12,.0f}"
                  f"{r['latency_median_ms']:>9.0f}ms")

        # ★★ 两把尺子。原来只印一把，而那把在比错的东西。
        #
        #   旧判据：「rerank 20→5 要在 CompRecall@5 上超过 baseline@10」
        #   —— 取的是 baseline@10 的**前 5 片子集**（0.741）。
        #   可 baseline@10 实际往 prompt 里送的是 10 片，它的操作性数字是
        #   CompRecall@**10**（0.914）。拿子集去比，等于给重排挑了一把宽松的尺子。
        #
        #   这正是探针十三留下的方法论：**两把尺子量同一件事，结论一致才下判断。**
        #   而这次是判据自己踩了坑。
        baseline, ce = results.get("baseline"), results.get("ce")
        print("\n  ★ 判据（两把尺子，都赢才算赢）")
        if baseline:
            print(f"    尺子一 · 等片数（都送 5 片）：要赢 baseline@5 的 "
                  f"CompRecall@5 = {baseline['complete_recall@5']:.3f}")
            print("             这一把量的是「重排把顺序排得更对了吗」")
            print(f"    尺子二 · CE 前 5 要追平 baseline 前 10 的 "
                  f"CompRecall@10 = {baseline['complete_recall@10']:.3f}"
                  f"（context {baseline['context_chars_median@10']:,.0f} 字符）")
        if ce:
            print(f"    CE 实测 · CompRecall@5 = {ce['complete_recall@5']:.3f}，"
                  f"context {ce['context_chars_median@5']:,.0f} 字符")
            print("             尺子二量的是「少送一半 context 还能不能一样好」")
        print("    只赢尺子一 → 重排有效但省不下成本；只赢尺子二 → 省了 context 但答得更差。")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "ablate_rerank_ce.json"
        out.write_text(json.dumps(
            {"corpus_files": total, "probes": len(probes),
             "ceilings": {str(k): v for k, v in caps.items()},
             "arms": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入 {out.relative_to(ROOT)}")
        return 0
    finally:
        if original_embed_query is not None:
            from minibrain import llamaindex_setup
            llamaindex_setup.embed_query = original_embed_query
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
