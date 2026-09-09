"""切分消融：`CHUNK_SIZE` 和 `CHUNK_OVERLAP` 该设多少，用数据定。

## 为什么做

`CHUNK_SIZE=800 / CHUNK_OVERLAP=120` 这两个数字**是拍脑袋定的**，从没验证过。
而「chunk 怎么切、切大切小各有什么问题、overlap 设多少」是 RAG 面试必问题
（见 docs/INTERVIEW.md 第一节，那 4 行现在全是「只能讲原理」）。

公开资料给的经验值是「中文 300~800 字、overlap 10~15%」。我们的 800/120 落在
范围内——但**「落在别人给的范围内」和「我测过」是两个答案**。

而且 chunk 大小会影响下游所有指标，**先把它定下来，后面的实验才有稳定地基**。

## ★★ 三个会毁掉这个实验的陷阱

### 陷阱一：索引膨胀（本项目刚栽过）

每换一个配置就要 purge + 重新入库一次。而反复 upload/purge 会让 HNSW 索引
累积死条目——实测曾涨到 1125MB（表仅 18MB），**优化器悄悄不用索引、召回也跟着掉，
两件事都不报错**（eval/RESULTS.md 探针十二）。

如果不管它，网格里**靠后的配置会被系统性地拖差**，而结论会写成
「chunk 越小越差」——完全是假的。
所以每个配置测量前都 `VACUUM + REINDEX + ANALYZE`，并把索引每行字节数打出来。

### 陷阱二：粒度混淆（这个最隐蔽）

评测指标是**按文件名**算的（必需文档有没有被召回）。但 chunk 越小，
同一个文件会切出越多片段，于是 **top-5 个片段可能全来自同一个文件**——
覆盖的**不同文件**反而更少。

这样比出来的「小 chunk 更差」不是检索更差，**是粒度不同导致的结构性劣势**。

所以本脚本用两把尺子量：
  1. **固定 k**：所有配置都取 top-5 片段（常规比法，但对小 chunk 不公平）
  2. **等字符预算**：都取到约 2400 字为止（**这才是公平的比法**——
     真正进 prompt 的是字符数，不是片段数）

两把尺子结论一致才敢下判断。

### 陷阱三：查询 embedding 的成本会淹没实验

每次 `search()` 都要为问题调一次 embedding（实测 1247ms，占端到端 92.4%）。
6 个配置 × 58 题 = 348 次，光这一项就要 7 分钟以上，而且**每次结果都一样**
（问题没变，只有语料切分方式变了）。

所以脚本里给 `embed_query` 套了一层记忆化。★ 这是**评测专用**的——
生产代码里没有这个缓存（那是待办事项，不是现状），别看到这里以为已经有了。

跑法：uv run --no-sync python scripts/ablate_chunking.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import sys
import time
import uuid

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ir_metrics import summarize                                  # noqa: E402
from minibrain import config as config_module                     # noqa: E402
from minibrain import gateway, identity                           # noqa: E402
from minibrain.db import close_all, vector_db                     # noqa: E402
from minibrain.modules.vector_rag import core, embeddings         # noqa: E402
from minibrain.scripts_purge import purge_user                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"
PROBES = ROOT / "eval" / "probes.json"
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"

# (chunk_size, overlap)。前四组固定 overlap ≈ 15% 扫大小，
# 后两组固定大小=800 隔离 overlap 的影响——**一次只动一个变量**。
GRID = [
    (300, 45),
    (500, 75),
    (800, 120),          # ← 当前默认值
    (1200, 180),
    (800, 0),            # overlap 归零：跨边界的语义会不会真的丢
    (800, 240),          # overlap 加倍：更多冗余换更少断裂，值不值
]

FIXED_K = 5
CHAR_BUDGET = 2400       # 等预算比较用。约等于 800 字 × 3 片段
TOP_K_FOR_BUDGET = 30    # 先多取一些，再按字符数截断


def memoize_embed_query() -> None:
    """给 embed_query 套记忆化。★ 只在本脚本内生效，生产没有缓存。"""
    cache: dict[str, list[float]] = {}
    original = embeddings.embed_query

    def cached(text: str):
        if text not in cache:
            cache[text] = original(text)
        return cache[text]

    embeddings.embed_query = cached
    core.embed_query = cached          # core 是 from ... import 进来的，要单独替换


def set_chunking(chunk_size: int, overlap: int) -> None:
    """改运行时配置。Config 是模块级单例，直接替换掉即可。"""
    cfg = config_module.get_config()
    config_module._config = dataclasses.replace(
        cfg, chunk_size=chunk_size, chunk_overlap=overlap)


def reindex() -> dict:
    """测量前清掉索引膨胀。见模块 docstring「陷阱一」。"""
    import psycopg
    from psycopg.rows import dict_row

    url = config_module.get_config().database_url
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO mod_vector, extensions")
            cur.execute("VACUUM (ANALYZE) chunks")
            cur.execute("REINDEX INDEX chunks_embedding_hnsw_idx")
            cur.execute("SELECT pg_relation_size('chunks_embedding_hnsw_idx') AS b, "
                        "count(*) AS n FROM chunks")
            row = cur.fetchone()
    return {"index_bytes_per_row": row["b"] / max(1, row["n"]), "chunks": row["n"]}


def ingest(user) -> tuple[int, float]:
    started = time.perf_counter()
    files = sorted(CORPUS.glob("*.md"))
    for path in files:
        gateway.process("vector-rag", gateway.call(
            "vector-rag", "upload_document", user, None,
            path.name, path.read_text(encoding="utf-8")))
    bad = [d for d in gateway.call("vector-rag", "list_documents", user)
           if d["status"] != "ready"]
    if bad:
        raise SystemExit(f"入库失败：{[(d['filename'], d['error']) for d in bad]}")
    return len(files), time.perf_counter() - started


def files_of(evidence) -> list[str]:
    seen, out = set(), []
    for e in evidence:
        name = e.location.split(" #")[0]
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def run_config(user, probes) -> dict:
    """同一批查询，两把尺子各量一次。"""
    fixed_pairs, budget_pairs = [], []
    fixed_files, fixed_chars, budget_chars, budget_chunks = [], [], [], []

    for p in probes:
        required = set(p["required"])

        # 尺子一：固定 k 个片段
        res = core.search(user, p["question"], top_k=FIXED_K)
        names = files_of(res.evidence)
        fixed_pairs.append((required, names))
        fixed_files.append(len(names))
        fixed_chars.append(sum(len(e.snippet) for e in res.evidence))

        # 尺子二：固定字符预算
        wide = core.search(user, p["question"], top_k=TOP_K_FOR_BUDGET)
        used, kept = 0, []
        for e in wide.evidence:
            if used + len(e.snippet) > CHAR_BUDGET and kept:
                break
            kept.append(e)
            used += len(e.snippet)
        budget_pairs.append((required, files_of(kept)))
        budget_chars.append(used)
        budget_chunks.append(len(kept))

    return {
        "fixed_k": summarize(fixed_pairs, (3, 5, 10)),
        "budget": summarize(budget_pairs, (3, 5, 10)),
        "fixed_distinct_files": statistics.mean(fixed_files),
        "fixed_context_chars": statistics.median(fixed_chars),
        "budget_context_chars": statistics.median(budget_chars),
        "budget_chunks": statistics.mean(budget_chunks),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="*", default=None,
                    help="只跑网格里的第几组（0-based），调试用")
    args = ap.parse_args()

    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))

    grid = GRID if args.grid is None else [GRID[i] for i in args.grid]
    memoize_embed_query()
    print(f"探针 {len(probes)} 题 | 网格 {len(grid)} 组 | "
          f"固定 k={FIXED_K} / 等预算 {CHAR_BUDGET} 字\n")

    rows = []
    for chunk_size, overlap in grid:
        set_chunking(chunk_size, overlap)
        tag = uuid.uuid4().hex[:6]
        user = identity.create_user(f"chunk_{tag}", "pw123456", is_admin=False)
        try:
            _files, secs = ingest(user)
            health = reindex()          # ★ 每组测量前都清膨胀，见陷阱一
            with vector_db() as cur:
                cur.execute("SELECT count(*) AS n, avg(length(content)) AS avg_len "
                            "FROM chunks")
                stat = cur.fetchone()
            result = run_config(user, probes)
            rows.append({
                "chunk_size": chunk_size, "overlap": overlap,
                "chunks": stat["n"], "avg_chunk_chars": float(stat["avg_len"] or 0),
                "ingest_seconds": secs, **health, **result,
            })
            print(f"  跑完 chunk_size={chunk_size} overlap={overlap} "
                  f"→ {stat['n']} 片段，入库 {secs:.0f}s")
        finally:
            purge_user(user)

    # ---------------- 报告 ----------------
    print("\n" + "=" * 100)
    print("尺子一：固定取 top-5 个片段（常规比法，但对小 chunk 不公平——见陷阱二）")
    print("=" * 100)
    print(f"  {'配置':<18}{'片段数':>8}{'平均字长':>10}{'CompRec@5':>12}"
          f"{'Recall@5':>11}{'NDCG@5':>10}{'MRR':>9}{'覆盖文件':>10}{'context字符':>13}")
    for r in rows:
        m = r["fixed_k"]
        print(f"  size={r['chunk_size']:<5} ov={r['overlap']:<5}"
              f"{r['chunks']:>8}{r['avg_chunk_chars']:>10.0f}"
              f"{m['complete_recall@5']:>12.3f}{m['recall@5']:>11.3f}"
              f"{m['ndcg@5']:>10.3f}{m['mrr']:>9.3f}"
              f"{r['fixed_distinct_files']:>10.1f}{r['fixed_context_chars']:>13,.0f}")

    print("\n" + "=" * 100)
    print(f"尺子二：固定 ~{CHAR_BUDGET} 字预算（公平比法——进 prompt 的是字符不是片段）")
    print("=" * 100)
    print(f"  {'配置':<18}{'取了几片':>10}{'实际字符':>11}{'CompRec@5':>12}"
          f"{'Recall@5':>11}{'NDCG@5':>10}{'MRR':>9}")
    for r in rows:
        m = r["budget"]
        print(f"  size={r['chunk_size']:<5} ov={r['overlap']:<5}"
              f"{r['budget_chunks']:>10.1f}{r['budget_context_chars']:>11,.0f}"
              f"{m['complete_recall@5']:>12.3f}{m['recall@5']:>11.3f}"
              f"{m['ndcg@5']:>10.3f}{m['mrr']:>9.3f}")

    print("\n  索引健康（每组测量前都 REINDEX 过，这一列应该都差不多）：")
    for r in rows:
        print(f"    size={r['chunk_size']:<5} ov={r['overlap']:<5}"
              f"索引每行 {r['index_bytes_per_row']:>7,.0f} 字节")
    print("    ★ 如果这一列差异很大，说明 REINDEX 没生效，上面所有结论作废。")

    print("\n  ★ 判读规矩：**两把尺子结论一致才下判断。**")
    print("    只在固定 k 下更好，很可能只是粒度带来的结构性优势，不是真的更好。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "ablate_chunking.json"
    out.write_text(json.dumps(
        {"probes": len(probes), "fixed_k": FIXED_K, "char_budget": CHAR_BUDGET,
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
