"""混合检索对比：向量 / 关键词 / 融合，同一套 58 题跑三遍。

目标（在动手之前就定死了，见 eval/RESULTS.md 探针一之三）：

  必须提上去              现在      目标
    会议编号 MRR          0.489    → 接近 1.000
    工单编号 MRR          0.708    → 接近 1.000
    项目编号 MRR          0.750    → 接近 1.000

  不许掉下来
    产品型号 / 英文缩写 / 罕见人名 MRR   1.000   → 保持
    原来那 10 道题 Recall@5              0.823   → 保持

**后一半和前一半同等重要。** 融合最典型的翻车方式是"修好一类、搞坏另一类"——
上一轮给 schema 注入列取值时就发生过（tbl-18 修好了，5 道题的路由被打回去）。

跑法：uv run --no-sync python scripts/probe_hybrid.py
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
PROBES_BLINDSPOT = ROOT / "eval" / "probes_blindspot.json"
OUT_DIR = ROOT / "eval" / "results"

MODES = ("vector", "keyword", "hybrid")
K_GRID = (3, 5, 10)


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


def retrieved_files(user, question: str, k: int, mode: str) -> list[str]:
    result = gateway.call("vector-rag", "search", user, question, top_k=k, mode=mode)
    seen, out = set(), []
    for e in result.evidence:
        name = e.location.split(" #")[0]
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def main() -> int:
    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    if PROBES_BLINDSPOT.is_file():
        probes += json.loads(PROBES_BLINDSPOT.read_text(encoding="utf-8"))

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"hyb_{tag}", "pw123456", is_admin=False)

    try:
        total = ingest(user)
        print(f"语料 {total} 篇 | 探针 {len(probes)} 题 | 模式 {'/'.join(MODES)}\n")

        # rankings[mode][probe_id] = 召回的文件名排序
        rankings: dict[str, dict[str, list[str]]] = {m: {} for m in MODES}
        for mode in MODES:
            for probe in probes:
                rankings[mode][probe["id"]] = retrieved_files(
                    user, probe["question"], total, mode)
            print(f"  {mode} 跑完")

        by_cat: dict[str, list[dict]] = defaultdict(list)
        for probe in probes:
            by_cat[probe["category"]].append(probe)

        def metrics(mode: str, subset: list[dict]) -> dict:
            return summarize(
                [(set(p["required"]), rankings[mode][p["id"]]) for p in subset], K_GRID)

        # ---------------- 主表：分类别 MRR ----------------
        print("\n" + "=" * 78)
        print("MRR（第一个相关文档排名的倒数；这是本次要改善的指标）")
        print("=" * 78)
        print(f"  {'类别':<20}{'向量':>10}{'关键词':>10}{'融合':>10}{'融合−向量':>12}")
        targets = []
        for cat in sorted(by_cat):
            m = {mode: metrics(mode, by_cat[cat]) for mode in MODES}
            delta = m["hybrid"]["mrr"] - m["vector"]["mrr"]
            flag = "  ← 目标" if cat.startswith("专有名词·") and "编号" in cat else ""
            print(f"  {cat:<20}{m['vector']['mrr']:>10.3f}{m['keyword']['mrr']:>10.3f}"
                  f"{m['hybrid']['mrr']:>10.3f}{delta:>+12.3f}{flag}")
            targets.append((cat, m["vector"]["mrr"], m["hybrid"]["mrr"]))

        # ---------------- 全局 ----------------
        print("\n" + "=" * 78)
        print("全局指标（全部 %d 题）" % len(probes))
        print("=" * 78)
        all_m = {mode: metrics(mode, probes) for mode in MODES}
        print(f"  {'指标':<20}{'向量':>10}{'关键词':>10}{'融合':>10}")
        for label, key in [("MRR", "mrr"), ("Recall@5", "recall@5"),
                           ("NDCG@5", "ndcg@5"), ("Hit Rate@5", "hit@5"),
                           ("Complete Recall@5", "complete_recall@5")]:
            print(f"  {label:<20}" + "".join(
                f"{all_m[mode][key]:>10.3f}" for mode in MODES))

        # ---------------- 回归检查 ----------------
        print("\n" + "=" * 78)
        print("回归检查：原本就满分的类别有没有被搞坏")
        print("=" * 78)
        broke = []
        for cat, before, after in targets:
            if before >= 0.999 and after < 0.999:
                broke.append((cat, before, after))
        if broke:
            for cat, before, after in broke:
                print(f"  ✗ {cat}  MRR {before:.3f} → {after:.3f}")
        else:
            print("  ✓ 没有类别从满分掉下来")

        orig = [p for p in probes if not p["category"].startswith("专有名词")]
        ov, oh = metrics("vector", orig), metrics("hybrid", orig)
        print(f"\n  原来那 10 道题：Recall@5 {ov['recall@5']:.3f} → {oh['recall@5']:.3f}"
              f"   NDCG@5 {ov['ndcg@5']:.3f} → {oh['ndcg@5']:.3f}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "probe_hybrid.json"
        out.write_text(json.dumps({
            "corpus_files": total, "probes": len(probes), "modes": list(MODES),
            "overall": all_m,
            "by_category": {cat: {m: metrics(m, by_cat[cat]) for m in MODES}
                            for cat in sorted(by_cat)},
            "rankings": rankings,
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
