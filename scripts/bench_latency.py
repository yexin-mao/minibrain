"""检索延迟基准。

## 为什么现在才做

这个项目有一大堆质量指标（Recall / MRR / NDCG / 溯源率 / 路由准确率），
**一个性能数字都没有**。而简历指南里 RAG 方向点名的量化指标是
「准确率、召回率、Top-K 命中率、**检索延迟**」——前三个有一堆，最后一个是零。

更实际的原因：**下一步要上 pgvector + HNSW 索引，那是拿召回率换速度的交换。
没有基线，就说不清"快了多少、召回掉了多少"。**

## 拆到阶段，不要只报一个总数

一次检索由几段完全不同的开销组成，混在一起报没法指导优化：

    SQL 拉取     把可见片段全量拉进内存        ← O(片段数)，随语料线性增长
    embedding    问题转向量，一次外部 API 调用  ← 网络往返，和语料量无关
    余弦计算     numpy 矩阵乘                  ← O(片段数 × 维度)
    BM25         纯 Python 分词 + 打分         ← O(片段数 × 文档长度)
    RRF          合并两个排名                  ← O(片段数)

**只有"和语料量相关"的那几段才是 pgvector 要解决的。**
embedding 那段换什么索引都省不掉——把它单独摘出来，才不会高估 ANN 的收益。

## 关于测量口径

- embedding 走网络，波动极大，所以单独报中位数和 p95，不和本地计算混在一起
- 本地计算部分预热一次再测（第一次会有 numpy 懒加载、CPU 缓存冷）
- 报中位数而不是平均：一次网络抖动就能把平均数拉飞

跑法：uv run --no-sync python scripts/bench_latency.py
      uv run --no-sync python scripts/bench_latency.py --repeat 20
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

from minibrain import gateway, identity                              # noqa: E402
from minibrain.db import close_all                                   # noqa: E402
from minibrain.modules.vector_rag.core import _visible_chunks        # noqa: E402
from minibrain.modules.vector_rag.embeddings import embed_query      # noqa: E402
from minibrain.modules.vector_rag.fusion import reciprocal_rank_fusion  # noqa: E402
from minibrain.modules.vector_rag.core import (                       # noqa: E402
    _keyword_ranking_indexed, _vector_search_in_db,
)
from minibrain.modules.vector_rag.keyword import rank_by_bm25        # noqa: E402
from minibrain.scripts_purge import purge_user                       # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"

QUERIES = [
    "二线城市住宿一晚最多报多少钱？",
    "MTG-20260617-02 这次会议的决议是什么？",
    "公司一共有几个部门，分别是什么？",
    "张敏的上级的上级是谁？",
    "X7-Pro 是什么产品？当前版本是多少？",
]


def timed(fn, repeat: int) -> tuple[list[float], object]:
    """跑 repeat 次，返回每次的毫秒数和最后一次的结果。已预热。"""
    fn()                                        # 预热：numpy 懒加载、CPU 缓存
    samples, result = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples, result


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"bench_{tag}", "pw123456", is_admin=False)

    try:
        files = sorted(CORPUS.glob("*.md"))
        for path in files:
            gateway.process("vector-rag", gateway.call(
                "vector-rag", "upload_document", user, None,
                path.name, path.read_text(encoding="utf-8")))

        rows = _visible_chunks(user)
        print(f"语料 {len(files)} 篇 / {len(rows)} 个片段 | 每项重复 {args.repeat} 次 | 报中位数\n")

        stages: dict[str, dict] = {}

        # ---- 1. SQL 拉取：把可见片段全量拉进内存 ----
        samples, _ = timed(lambda: _visible_chunks(user), args.repeat)
        stages["SQL 拉取全部片段"] = summarize(samples)

        contents = [r["content"] for r in rows]

        # ---- 2. embedding：一次外部 API 调用 ----
        # 只跑一条查询、次数减半——它是网络往返，跑多了纯粹烧钱且波动大
        emb_samples, _ = timed(lambda: embed_query(QUERIES[0]), max(3, args.repeat // 2))
        stages["embedding（外部 API）"] = summarize(emb_samples)

        # ---- 3. 向量检索（现在在数据库里做）----
        # 原来这里是「把 900×1024 浮点数搬进内存 + numpy 算余弦」，
        # 现在是「数据库用 HNSW 索引排完序，只发回 top-5」。
        # 注意它含 embedding 的网络往返，所以要减掉才是纯检索开销。
        samples, _ = timed(
            lambda: [_vector_search_in_db(user, q, 5) for q in QUERIES], args.repeat)
        stages["向量检索 pgvector（5 条，含 embedding）"] = summarize(samples)

        # ---- 4. BM25 ----
        # 两个版本都测：内存版（每次重新分词）vs 倒排索引版（查表）。
        # 这是本次优化的直接对照，而且质量指标必须一位小数都不变。
        samples, _ = timed(
            lambda: [rank_by_bm25(q, contents) for q in QUERIES], args.repeat)
        stages["BM25 内存版（5 条）"] = summarize(samples)

        samples, _ = timed(
            lambda: [_keyword_ranking_indexed(user, q, rows) for q in QUERIES], args.repeat)
        stages["BM25 倒排索引版（5 条）"] = summarize(samples)

        # ---- 5. RRF ----
        kr = [_keyword_ranking_indexed(user, q, rows) for q in QUERIES]
        vr = [list(range(len(kr[i]))) for i in range(len(QUERIES))]   # RRF 只关心排名长度
        samples, _ = timed(
            lambda: [reciprocal_rank_fusion([v, k], tie_breaker=k)
                     for v, k in zip(vr, kr)], args.repeat)
        stages["RRF 融合（5 条查询）"] = summarize(samples)

        # ---- 端到端 ----
        samples, _ = timed(
            lambda: gateway.search("vector-rag", user, QUERIES[0], top_k=5), args.repeat)
        stages["端到端 search()"] = summarize(samples)

        print("=" * 74)
        print(f"  {'阶段':<26}{'中位数':>10}{'p95':>10}{'最小':>10}{'最大':>10}")
        print("=" * 74)
        for name, s in stages.items():
            print(f"  {name:<26}{s['median']:>9.2f}ms{s['p95']:>9.2f}ms"
                  f"{s['min']:>9.2f}ms{s['max']:>9.2f}ms")

        # ---- 拆解：本地计算 vs 网络 ----
        per_query = {
            "SQL 拉取": stages["SQL 拉取全部片段"]["median"],
            "向量检索(减去embedding)": max(
                0.0,
                stages["向量检索 pgvector（5 条，含 embedding）"]["median"] / len(QUERIES)
                - stages["embedding（外部 API）"]["median"]),
            "BM25": stages["BM25 倒排索引版（5 条）"]["median"] / len(QUERIES),
            "RRF": stages["RRF 融合（5 条查询）"]["median"] / len(QUERIES),
        }
        local = sum(per_query.values())
        network = stages["embedding（外部 API）"]["median"]

        print("\n" + "=" * 74)
        print("单次检索的构成（中位数）")
        print("=" * 74)
        for name, ms in per_query.items():
            print(f"  {name:<26}{ms:>9.2f}ms   {ms / (local + network) * 100:>5.1f}%")
        print(f"  {'embedding（网络）':<26}{network:>9.2f}ms   "
              f"{network / (local + network) * 100:>5.1f}%")
        print(f"  {'—' * 24}")
        print(f"  {'本地计算合计':<26}{local:>9.2f}ms   "
              f"{local / (local + network) * 100:>5.1f}%")
        print(f"  {'合计':<26}{local + network:>9.2f}ms")

        print("\n  ★ 只有「本地计算」那部分随语料量增长，也只有它是 pgvector 能优化的。")
        print("    embedding 是网络往返，换什么索引都省不掉——单独摘出来才不会高估 ANN 的收益。")
        print(f"    当前 {len(rows)} 个片段下，本地计算只占 {local / (local + network) * 100:.0f}%，")
        print("    所以现在换 pgvector **不会**让用户感觉更快。它的价值要在片段数大得多时才显现。")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "bench_latency.json"
        out.write_text(json.dumps({
            "corpus_files": len(files), "chunks": len(rows), "repeat": args.repeat,
            "stages": stages, "per_query_ms": per_query,
            "local_total_ms": local, "network_ms": network,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入 {out.relative_to(ROOT)}")
        return 0

    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
