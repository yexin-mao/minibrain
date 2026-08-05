"""HNSW 召回损失评测：近似检索到底丢了多少。

## 为什么这个脚本是 pgvector 这一步的核心

**HNSW 是近似最近邻算法——它用召回率换速度。**

绝大多数项目加了向量数据库只能说"我用了 HNSW"，**说不出损失了多少召回**，
因为没有标注评测集。本项目有 58 道题，正好能把这个交换量出来。

## 测什么

同一批查询、同一份数据，只换检索方式：

★★ 测量方法（踩过坑，见下方代码注释）：**预热 + 打乱顺序交错测量**。
   第一版按 ef 从小到大依次测、不预热，测出 ef=20 比 ef=10 还快 ——
   违反理论。根因是第一个被测的配置独占了冷缓存代价。
   **基准测试里顺序本身就是一个变量。**

  精确检索   SET enable_indexscan = off  → 全表扫描，保证拿到真正的最近邻
  HNSW      SET enable_seqscan = off    → 试图走索引，ef_search 从小到大扫一遍

★★ 两边都要**显式强制**，而且**强制完必须用 EXPLAIN 核对**（见 assert_plans）。
   默认让优化器自己选的话，它在这个规模下两次都会选全表扫描，
   于是"对照实验"变成自己和自己比 —— 这个坑本项目踩过**两次**，
   第二次是因为第一次的教训只写成了注释，没写成检查。

对每个配置报：**延迟** + **Recall@k / MRR / NDCG**。

★ 还额外报一个 IR 里没有的指标：**Top-k 一致率**
  ——HNSW 返回的 top-k 和精确检索的 top-k 重合多少。
  它比 Recall 更直接地衡量"近似"本身的代价：
  Recall 只看标注的相关文档，而一致率看的是"和真值排序差多少"，
  跟标注质量无关。

## ef_search 是什么

HNSW 查询时的候选集大小。**这是近似检索唯一的运行时旋钮**：
大 → 搜得更广，召回高、延迟高；小 → 快但可能漏。pgvector 默认 40。

建索引的参数（m、ef_construction）改了要重建索引，不在本脚本的扫描范围内。

跑法：uv run --no-sync python scripts/probe_hnsw.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time
import uuid

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ir_metrics import summarize                                  # noqa: E402
from minibrain import gateway, identity                           # noqa: E402
from minibrain.db import close_all, vector_db                     # noqa: E402
from minibrain.modules.vector_rag.core import (                   # noqa: E402
    _to_vector_literal, _visibility_clause,
)
from minibrain.modules.vector_rag.embeddings import embed_query    # noqa: E402
from minibrain.scripts_purge import purge_user                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"
PROBES = ROOT / "eval" / "probes.json"
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"

EF_GRID = (10, 20, 40, 100, 200)
K = 10


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


SQL = """
    SELECT c.id, d.filename
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    JOIN sources s ON s.id = c.source_id
    WHERE {where} AND d.status = 'ready'
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s
"""


def _apply_mode(cur, *, exact: bool, ef: int) -> None:
    if exact:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
    else:
        # ★ 显式关掉全表扫描，试图让优化器走 HNSW。
        #   注意「试图」——这行**不保证**成功，见 assert_plans() 的说明。
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")


def check_plans(user, literal: str) -> dict:
    """确认每个配置**实际**走了什么计划。返回 {配置名: "HNSW" | "全扫描"}。

    ★★ 这个函数的存在本身就是一条教训，而且是踩了三次才写出来的。

    第 1 次：「精确」和「HNSW」两组**都走了全表扫描**——对照实验变成自己和
    自己比，输出却完全正常（只是所有配置数字一样）。修法是加一行
    `SET enable_seqscan = off`，**并把教训写进注释**。

    第 2 次：同样的症状复发。关掉 Seq Scan 只关掉了一种走法，PostgreSQL 转头
    用 `chunks_source_idx`（source_id 上的普通 btree）做全扫描 + 排序，
    HNSW 照样没走上。→ 教训写进注释拦不住复发，得写成代码里的检查。

    第 3 次：加了检查之后，检查**通过了**，测出来的数字却依然全都一样。
    因为检查跑在灌完语料之后、正式测量之前，而中间隔着 58 次 embedding
    网络调用（约 1 分钟）——**autoanalyze 在这段时间里更新了统计信息，
    计划在测量开始前就已经变了**。核对的是一个已经过期的计划。

    所以现在三条一起做：
      1. 测量前显式 `ANALYZE chunks`，不等 autoanalyze 在半路插进来
      2. 用**真实的查询向量**核对，不用零向量
      3. **逐个 ef 核对**，因为 ef_search 会进入 pgvector 的代价估算，
         同一条 SQL 在不同 ef 下可能走不同计划

    ★ 另一个反复咬人的坑：**HNSW 索引会因为反复 upload/purge 而膨胀**。
      开发过程中它一度涨到 1125MB（表本身只有 18MB），膨胀之后优化器就不选它了，
      **不报错，只是悄悄退化成全表扫描**。
      `REINDEX INDEX chunks_embedding_hnsw_idx` 之后立刻恢复。
      所以下面把索引/表的大小也打出来——测出「索引莫名其妙不被使用」时先看这个。
    """
    where, params = _visibility_clause(user)
    plans = {}
    with vector_db() as cur:
        cur.execute("ANALYZE chunks")
        for label, exact, ef in ([("精确", True, 40)]
                                 + [(f"ef={e}", False, e) for e in EF_GRID]):
            _apply_mode(cur, exact=exact, ef=ef)
            cur.execute("EXPLAIN (COSTS OFF) " + SQL.format(where=where),
                        params + [literal, K])
            plan = " ".join(str(next(iter(r.values()))) for r in cur.fetchall())
            plans[label] = "HNSW" if "chunks_embedding_hnsw_idx" in plan else "全扫描"

        # ★ 判断膨胀**不能**比「索引 vs 表」。pgvector 的索引里存着向量副本，
        #   而表里的向量被 TOAST 到行外，pg_relation_size('chunks') 不算它——
        #   所以索引比表大好几倍是**正常的**，拿它当膨胀信号会误判。
        #   正确的基准是「每行占多少字节」：一行至少要存 dim×4 字节的向量，
        #   加上图的连接边，健康值大约是它的 1.5~2 倍。
        cur.execute("SELECT pg_relation_size('chunks_embedding_hnsw_idx') AS b, "
                    "count(*) AS n FROM chunks")
        row = cur.fetchone()
        per_row = row["b"] / max(1, row["n"])
        expected = 1024 * 4 * 1.5

    print("  计划核对：" + "  ".join(f"{k}={v}" for k, v in plans.items()))
    verdict = "正常" if per_row < expected * 2 else "★ 偏大，疑似膨胀，建议 REINDEX"
    print(f"  索引每行 {per_row:,.0f} 字节（健康约 {expected:,.0f}）→ {verdict}")

    if plans["精确"] == "HNSW":
        raise SystemExit("「精确」组走了索引，拿不到真值，已中止。")
    if all(v == "全扫描" for k, v in plans.items() if k != "精确"):
        raise SystemExit(
            "所有 HNSW 配置都没走索引，测出来会和精确组一模一样，已中止。\n"
            "排查顺序：① 索引每行字节数（见上面）是否偏大 → REINDEX\n"
            "          ② 片段数是否太少（实测 210 不走、900 走）")
    return plans


def retrieve(user, literal: str, k: int, *, exact: bool, ef: int = 40):
    """返回 (文件名排序, 耗时毫秒)。exact=True 强制全表扫描做真值。"""
    where, params = _visibility_clause(user)
    start = time.perf_counter()
    with vector_db() as cur:
        _apply_mode(cur, exact=exact, ef=ef)
        cur.execute(SQL.format(where=where), params + [literal, k])
        rows = cur.fetchall()
    elapsed = (time.perf_counter() - start) * 1000

    seen, files = set(), []
    for r in rows:
        if r["filename"] not in seen:
            seen.add(r["filename"])
            files.append(r["filename"])
    return files, elapsed, [str(r["id"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=200,
                    help="正式测量前空跑多少次，让缓存和 JIT 进入稳态")
    args = ap.parse_args()

    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"hnsw_{tag}", "pw123456", is_admin=False)

    try:
        total = ingest(user)
        with vector_db() as cur:
            cur.execute("SELECT count(*) AS n FROM chunks")
            chunks = cur.fetchone()["n"]
        print(f"语料 {total} 篇 / {chunks} 片段 | 探针 {len(probes)} 题 | top-{K}\n")

        # 查询向量只算一次，后面所有配置复用——否则测的就是 embedding API 的波动
        literals = {p["id"]: _to_vector_literal(embed_query(p["question"])) for p in probes}
        print("查询向量已缓存，下面的延迟不含 embedding 网络往返\n")

        # ★ 核对计划必须放在 embedding 之后、测量之前。
        #   放在前面的话，中间那一分钟的网络调用足够让 autoanalyze 改掉计划，
        #   核对结果在测量开始时就已经过期了——这个坑踩过，见 check_plans 文档。
        plans = check_plans(user, literals[probes[0]["id"]])
        print()

        results: dict[str, dict] = {}

        # ---- 精确检索：真值 ----
        for _ in range(args.warmup):
            retrieve(user, literals[probes[0]["id"]], K, exact=True)

        exact_files, exact_ids, exact_lat = {}, {}, []
        for p in probes:
            files, _ms, ids = retrieve(user, literals[p["id"]], K, exact=True)
            exact_files[p["id"]], exact_ids[p["id"]] = files, ids
        for _ in range(args.repeat):
            for p in probes[:5]:
                _f, ms, _i = retrieve(user, literals[p["id"]], K, exact=True)
                exact_lat.append(ms)
        m = summarize([(set(p["required"]), exact_files[p["id"]]) for p in probes], (5, 10))
        results["精确（全表扫描）"] = {
            "latency_median": statistics.median(exact_lat),
            "latency_stdev": statistics.stdev(exact_lat),
            "agreement": 1.0, "plan": plans["精确"], **m,
        }

        # ---- HNSW：扫 ef_search ----
        #
        # ★★ 测量方法上踩过一次，教训写在这里，别再犯：
        #
        # 第一版是「按 ef 从小到大依次测，每个重复 5 次，不预热」，结果测出
        #     ef=10  0.89ms
        #     ef=20  0.80ms   ← 比 ef=10 还快
        # 这**违反理论**——ef 越大搜得越广，只可能更慢。
        #
        # 根因是两个实验设计错误：
        #   ① 没预热 → 第一个被测的配置（ef=10）独自承担了冷缓存的代价
        #   ② 顺序执行 → 后面的配置白白享受前面暖好的缓存
        # 而且当时的标准差（0.183ms vs 后面几档的 0.04ms）本身就是信号，我没看出来。
        #
        # 加了预热 + 打乱顺序交错测量之后，曲线立刻变成单调的：
        #     ef=10 0.369 / 20 0.401 / 40 0.538 / 100 0.720 / 200 0.844 / 400 0.989ms
        #
        # 教训：**基准测试里，顺序本身就是一个变量。** 交错测量才能把它消掉。
        for _ in range(args.warmup):
            retrieve(user, literals[probes[0]["id"]], K, exact=False, ef=40)

        # 先把召回类指标测出来（和延迟无关，跑一遍就够）
        files_by = {}
        for ef in EF_GRID:
            files_by[ef] = {}
            for p in probes:
                files, _ms, ids = retrieve(user, literals[p["id"]], K, exact=False, ef=ef)
                truth = exact_ids[p["id"]]
                files_by[ef][p["id"]] = (files, len(set(ids) & set(truth)) / max(1, len(truth)))

        # 延迟单独测，打乱顺序交错
        latencies = {ef: [] for ef in EF_GRID}
        schedule = [ef for ef in EF_GRID for _ in range(args.repeat)]
        random.Random(20260806).shuffle(schedule)
        for ef in schedule:
            for p in probes[:5]:                       # 固定 5 条查询，避免题目差异混进延迟
                _f, ms, _i = retrieve(user, literals[p["id"]], K, exact=False, ef=ef)
                latencies[ef].append(ms)

        for ef in EF_GRID:
            m = summarize([(set(p["required"]), files_by[ef][p["id"]][0]) for p in probes], (5, 10))
            agreements = [files_by[ef][p["id"]][1] for p in probes]
            results[f"HNSW ef_search={ef}"] = {
                "latency_median": statistics.median(latencies[ef]),
                "latency_stdev": statistics.stdev(latencies[ef]),
                "agreement": sum(agreements) / len(agreements),
                "plan": plans[f"ef={ef}"], **m,
            }

        # ---- 报告 ----
        print("=" * 84)
        print("HNSW 是近似算法：拿召回换速度。这张表就是那个交换的价格")
        print("=" * 84)
        print(f"  {'配置':<22}{'实走计划':>9}{'延迟中位数':>12}{'标准差':>10}"
              f"{'Top-k一致率':>13}{'Recall@5':>11}{'MRR':>9}{'NDCG@5':>10}")
        base = results["精确（全表扫描）"]
        for name, r in results.items():
            speed = f"  ({base['latency_median'] / r['latency_median']:.1f}x)" if r is not base else ""
            print(f"  {name:<22}{r['plan']:>9}{r['latency_median']:>10.3f}ms"
                  f"{r['latency_stdev']:>8.3f}ms{r['agreement']:>12.1%}"
                  f"{r['recall@5']:>11.3f}{r['mrr']:>9.3f}{r['ndcg@5']:>10.3f}{speed}")

        print("\n  ★ Top-k 一致率 = HNSW 返回的 top-10 和精确检索的 top-10 重合多少。")
        print("    它比 Recall 更直接地衡量「近似」本身的代价——Recall 只看标注的相关文档，")
        print("    一致率看的是和真值排序差多少，跟标注质量无关。")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "probe_hnsw.json"
        out.write_text(json.dumps({
            "corpus_files": total, "chunks": chunks, "probes": len(probes),
            "top_k": K, "repeat": args.repeat, "ef_grid": list(EF_GRID),
            "results": results,
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
