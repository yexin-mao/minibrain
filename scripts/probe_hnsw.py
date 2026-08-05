"""HNSW 召回损失评测：近似检索到底丢了多少。

## 为什么这个脚本是 pgvector 这一步的核心

**HNSW 是近似最近邻算法——它用召回率换速度。**

绝大多数项目加了向量数据库只能说"我用了 HNSW"，**说不出损失了多少召回**，
因为没有标注评测集。本项目有 58 道题，正好能把这个交换量出来。

## 测什么

同一批查询、同一份数据，只换检索方式：

  精确检索   SET enable_indexscan = off  → 全表扫描，保证拿到真正的最近邻
  HNSW      SET enable_seqscan = off    → 强制走索引，ef_search 从小到大扫一遍

★ 两边都要**显式强制**。默认让优化器自己选的话，它在这个规模下两次都会选全表扫描
  （成本模型估算 HNSW 贵 40 倍，实测反而快 3.2 倍），于是"对照实验"变成了自己和自己比。

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


def retrieve(user, literal: str, k: int, *, exact: bool, ef: int = 40):
    """返回 (文件名排序, 耗时毫秒)。exact=True 强制全表扫描做真值。"""
    where, params = _visibility_clause(user)
    start = time.perf_counter()
    with vector_db() as cur:
        if exact:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_bitmapscan = off")
        else:
            # ★ 必须显式关掉全表扫描，否则优化器根本不会用 HNSW 索引。
            #
            # 实测（EXPLAIN）：490 片段时优化器估算
            #     全表扫描 cost 65  vs  HNSW 索引 cost 2614
            # 于是它选全表扫描。但实际跑下来 HNSW 快 3.2 倍（0.86ms vs 2.73ms）——
            # **成本模型在这个规模下估错了 40 倍**。
            #
            # 第一版没加这两行，结果"精确"和"HNSW"两组**都走了全表扫描**，
            # 测出来所有配置一模一样。那是本项目第五次「指标先于系统出错」。
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")
        cur.execute(
            f"""
            SELECT c.id, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN sources s ON s.id = c.source_id
            WHERE {where} AND d.status = 'ready'
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            params + [literal, k],
        )
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
    ap.add_argument("--repeat", type=int, default=5)
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

        results: dict[str, dict] = {}

        # ---- 精确检索：真值 ----
        exact_files, exact_ids, latencies = {}, {}, []
        for p in probes:
            for _ in range(args.repeat):
                files, ms, ids = retrieve(user, literals[p["id"]], K, exact=True)
                latencies.append(ms)
            exact_files[p["id"]], exact_ids[p["id"]] = files, ids
        m = summarize([(set(p["required"]), exact_files[p["id"]]) for p in probes], (5, 10))
        results["精确（全表扫描）"] = {
            "latency_median": statistics.median(latencies),
            "agreement": 1.0, **m,
        }

        # ---- HNSW：扫 ef_search ----
        for ef in EF_GRID:
            files_by, latencies, agreements = {}, [], []
            for p in probes:
                for _ in range(args.repeat):
                    files, ms, ids = retrieve(user, literals[p["id"]], K, exact=False, ef=ef)
                    latencies.append(ms)
                files_by[p["id"]] = files
                # Top-k 一致率：和精确检索返回的 chunk id 重合多少
                truth = exact_ids[p["id"]]
                agreements.append(len(set(ids) & set(truth)) / max(1, len(truth)))
            m = summarize([(set(p["required"]), files_by[p["id"]]) for p in probes], (5, 10))
            results[f"HNSW ef_search={ef}"] = {
                "latency_median": statistics.median(latencies),
                "agreement": sum(agreements) / len(agreements), **m,
            }

        # ---- 报告 ----
        print("=" * 84)
        print("HNSW 是近似算法：拿召回换速度。这张表就是那个交换的价格")
        print("=" * 84)
        print(f"  {'配置':<22}{'延迟中位数':>12}{'Top-k一致率':>13}"
              f"{'Recall@5':>11}{'MRR':>9}{'NDCG@5':>10}")
        base = results["精确（全表扫描）"]
        for name, r in results.items():
            speed = f"  ({base['latency_median'] / r['latency_median']:.1f}x)" if r is not base else ""
            print(f"  {name:<22}{r['latency_median']:>10.2f}ms{r['agreement']:>12.1%}"
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
