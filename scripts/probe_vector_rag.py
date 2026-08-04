"""探针：向量检索链路在什么问题上会失效。

不是为了黑 Traditional RAG —— 对照组就是用来防止把它讲成稻草人的。
目的是拿到客观数据，回答两个问题：
  1. 它在哪类问题上稳定失效？
  2. 单纯调大 top_k 能不能解决？（能的话就不需要换范式）

判定标准是可计算的：每题预先标注「答对必须召回哪几篇」，
不需要人肉看效果，不需要 LLM 打分。

指标用 IR 领域的标准定义（Recall@k / Precision@k / Hit@k / MRR / NDCG@k，
见 scripts/ir_metrics.py），外加一个明确标注为非标准的严格变体
Complete Recall@k —— 因为多跳和全局聚合类问题漏一篇答案就是错的。

跑法：uv run --no-sync python scripts/probe_vector_rag.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ir_metrics import summarize                 # noqa: E402
from minibrain import gateway, identity          # noqa: E402
from minibrain.db import close_all               # noqa: E402
from minibrain.scripts_purge import purge_user   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
PROBES = ROOT / "eval" / "probes.json"
# 盲区探针由 gen_corpus.py 从 blindspots.json 生成。分开放是为了区分
# "人工设计的题" 和 "由植入词自动生成的题"——后者的价值在于覆盖面，
# 前者的价值在于每道题背后有具体的失效假设。
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"
OUT_DIR = ROOT / "eval" / "results"
K_GRID = (3, 5, 10)


def ingest(user) -> int:
    files = sorted(CORPUS.glob("*.md"))
    for path in files:
        doc_id = gateway.call(
            "vector-rag", "upload_document", user, None, path.name,
            path.read_text(encoding="utf-8"),
        )
        gateway.process("vector-rag", doc_id)

    docs = gateway.call("vector-rag", "list_documents", user)
    bad = [d for d in docs if d["status"] != "ready"]
    if bad:
        raise SystemExit(f"入库失败：{[(d['filename'], d['error']) for d in bad]}")
    return len(files)


def retrieved_files(user, question: str, k: int) -> list[str]:
    """location 形如 'dept-tech.md #0'，取文件名并保序去重。"""
    result = gateway.search("vector-rag", user, question, top_k=k)
    seen, out = set(), []
    for e in result.evidence:
        name = e.location.split(" #")[0]
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def main() -> int:
    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"probe_{tag}", "pw123456", is_admin=False)

    try:
        total = ingest(user)
        probes = json.loads(PROBES.read_text(encoding="utf-8"))
        if PROBES_BLINDSPOT.is_file():
            probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))
        print(f"语料 {total} 篇，探针 {len(probes)} 题\n")

        by_cat: dict[str, list] = defaultdict(list)
        details = []

        for probe in probes:
            required = set(probe["required"])
            # 一次取全量排序，再按 k 切片 —— 与分别检索完全等价，但省掉重复的 query embedding
            ranking = retrieved_files(user, probe["question"], total)

            row = {"probe": probe, "ranking": ranking, "at_k": {}}
            for k in K_GRID:
                hit = required & set(ranking[:k])
                row["at_k"][k] = {
                    "hit": len(hit),
                    "need": len(required),
                    "answerable": required <= set(ranking[:k]),
                }

            # 最小可答 k：要把必需文档全部捞齐，top_k 得开到多大
            positions = [ranking.index(f) + 1 for f in required if f in ranking]
            row["min_k"] = max(positions) if len(positions) == len(required) else None

            by_cat[probe["category"]].append(row)
            details.append(row)

        # ---------------- 明细 ----------------
        for cat, rows in by_cat.items():
            print("=" * 72)
            print(cat)
            print("=" * 72)
            for row in rows:
                p = row["probe"]
                marks = "  ".join(
                    f"k={k}: {row['at_k'][k]['hit']}/{row['at_k'][k]['need']}"
                    f"{'✓' if row['at_k'][k]['answerable'] else '✗'}"
                    for k in K_GRID
                )
                mk = row["min_k"]
                print(f"\n  {p['question']}")
                print(f"    {marks}    最小可答 k = {mk if mk else '不可达'}")
                if not row["at_k"][5]["answerable"]:
                    missing = set(p["required"]) - set(row["ranking"][:5])
                    print(f"    k=5 时漏掉：{', '.join(sorted(missing))}")
                    print(f"    实际召回前5：{', '.join(row['ranking'][:5])}")
            print()

        # ---------------- 汇总 ----------------
        print("=" * 72)
        print("汇总一：各类问题的 Complete Recall@k（必需文档全部召回才算）")
        print("=" * 72)
        print(f"{'类别':<18}" + "".join(f"{'k='+str(k):>10}" for k in K_GRID))
        for cat, rows in by_cat.items():
            cells = ""
            for k in K_GRID:
                ok = sum(1 for r in rows if r["at_k"][k]["answerable"])
                cells += f"{ok}/{len(rows):<8}".rjust(10)
            print(f"{cat:<18}{cells}")

        print(f"\n（语料共 {total} 篇。最小可答 k 接近 {total} 意味着检索基本没起作用——"
              f"等于把整个语料塞进上下文）")

        # ---------------- 标准 IR 指标 ----------------
        pairs = [(set(r["probe"]["required"]), r["ranking"]) for r in details]
        overall = summarize(pairs, K_GRID)

        print("\n" + "=" * 72)
        print(f"汇总二：标准 IR 指标（全部 {len(details)} 题平均）")
        print("=" * 72)
        print(f"  MRR = {overall['mrr']:.3f}   （第一个相关文档排名的倒数，越接近 1 越好）\n")
        print(f"  {'':<20}" + "".join(f"{'k='+str(k):>12}" for k in K_GRID))
        for name, key in [
            ("Recall@k", "recall"),
            ("Precision@k", "precision"),
            ("Hit Rate@k", "hit"),
            ("NDCG@k", "ndcg"),
            ("Complete Recall@k", "complete_recall"),
        ]:
            cells = "".join(f"{overall[f'{key}@{k}']:>12.3f}" for k in K_GRID)
            print(f"  {name:<20}{cells}")

        print("\n  怎么读这张表：")
        print("  - Recall 高而 Complete Recall 低 → 每题都召回了一部分，但很少召回全；")
        print("    对「公司有几个部门」这种题，召回 4/5 篇 = 答案错。")
        print("  - Precision 随 k 增大而下降是正常的：多数题只有 1~2 篇相关文档，")
        print("    k=10 时 Precision 上限就只有 0.1~0.2。这个指标在本场景下参考价值有限。")
        print("  - NDCG 比 Recall 多看一件事：相关文档排在第几位。")

        by_cat_metrics = {
            cat: summarize(
                [(set(r["probe"]["required"]), r["ranking"]) for r in rows], K_GRID
            )
            for cat, rows in by_cat.items()
        }

        print("\n" + "=" * 72)
        print("汇总三：分类别的 Recall@5 / NDCG@5 / MRR")
        print("=" * 72)
        print(f"  {'类别':<18}{'Recall@5':>12}{'NDCG@5':>10}{'MRR':>8}")
        for cat, m in by_cat_metrics.items():
            print(f"  {cat:<18}{m['recall@5']:>12.3f}{m['ndcg@5']:>10.3f}{m['mrr']:>8.3f}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "probe_vector_rag.json"
        out.write_text(json.dumps({
            "corpus_files": total,
            "probes": len(details),
            "k_grid": list(K_GRID),
            "overall": overall,
            "by_category": by_cat_metrics,
            "per_probe": [
                {
                    "id": r["probe"]["id"],
                    "category": r["probe"]["category"],
                    "question": r["probe"]["question"],
                    "required": r["probe"]["required"],
                    "ranking": r["ranking"],
                    "min_k": r["min_k"],
                }
                for r in details
            ],
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
