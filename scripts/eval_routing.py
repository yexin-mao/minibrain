"""路由评测：Agent 到底会不会自己选对链路？

这是本项目核心主张的唯一量化验证。之前所有"两条链路不能合并"的论据，
都建立在一个前提上：**Agent 面对一个问题时，知道该走哪一条**。
这个前提在这个脚本之前从来没被测过。

指标定义（都不需要 LLM 打分，可复现）：
  严格路由准确率  called == expected      调对了，且没多调
  宽松路由准确率  called ⊇ expected       该调的都调了（可能有多余调用）
  过度调用率      called ⊋ expected       多调了工具，浪费 token 和延迟
  答案正确率      预标答案出现在最终回答里（数字归一化后做子串匹配）

答案正确率是次要指标：它会被表述方式影响（"305.3万" vs "3053000"），
所以每题可以给多个可接受写法。路由准确率才是主指标——它是离散的、没有歧义。

跑法：
  uv run --no-sync python scripts/eval_routing.py
  uv run --no-sync python scripts/eval_routing.py --limit 5        # 快速冒烟
  uv run --no-sync python scripts/eval_routing.py --only tbl-09    # 单题排查
  uv run --no-sync python scripts/eval_routing.py --workers 1      # 串行，便于看日志
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")

from minibrain import gateway, identity          # noqa: E402
from minibrain.handwritten.agent_loop import answer          # noqa: E402
from minibrain.config import get_config          # noqa: E402
from minibrain.contracts import ModuleError      # noqa: E402
from minibrain.db import close_all               # noqa: E402
from minibrain.scripts_purge import purge_user   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_DOC = ROOT / "eval" / "corpus"
CORPUS_TBL = ROOT / "eval" / "corpus_table"
GOLDEN = ROOT / "eval" / "routing.json"
OUT_DIR = ROOT / "eval" / "results"

# 认定模型"承认查不到"的措辞。刻意收窄：只认明确的否定，
# 不把"根据检索到的信息"这种铺垫算进来——否则 edge 题会假阳性。
REFUSAL_MARKERS = (
    "没有查到", "未查到", "没有找到", "未找到", "查不到", "没有相关",
    "无相关", "不包含", "没有涉及", "未涉及", "无法回答", "没有提供",
    "未提供", "不支持", "无权", "只读",
)


def normalize(text: str) -> str:
    """数字归一化：去掉千分位和空白，让 '3,053,000' 和 '3053000' 可比。"""
    return re.sub(r"[,\s，、]", "", text)


def ingest(user) -> tuple[int, int]:
    """两条链路各自灌数据。同一家虚构公司，一半在文档一半在表——这是路由有意义的前提。"""
    docs = sorted(CORPUS_DOC.glob("*.md"))
    for path in docs:
        doc_id = gateway.call(
            "vector-rag", "upload_document", user, None, path.name,
            path.read_text(encoding="utf-8"),
        )
        gateway.process("vector-rag", doc_id)

    tables = sorted(CORPUS_TBL.glob("*.csv"))
    for path in tables:
        ds_id = gateway.call(
            "table-rag", "upload_csv", user, None, path.name, path.read_bytes()
        )
        gateway.process("table-rag", ds_id)

    bad_docs = [d for d in gateway.call("vector-rag", "list_documents", user)
                if d["status"] != "ready"]
    bad_tbls = [d for d in gateway.call("table-rag", "list_datasets", user)
                if d["status"] != "ready"]
    if bad_docs or bad_tbls:
        raise SystemExit(
            f"入库失败：docs={[(d['filename'], d['error']) for d in bad_docs]} "
            f"tables={[(d['filename'], d['error']) for d in bad_tbls]}"
        )
    return len(docs), len(tables)


def run_case(user, case: dict) -> dict:
    started = time.monotonic()
    try:
        result = answer(user, case["question"])
        text = result.answer
        called = [t.name for t in result.trace]
        sqls = [t.arguments for t in result.trace if t.name == "table_query"]
        error = None
    except ModuleError as exc:
        text, called, sqls, error = "", [], [], f"{exc.code}: {exc.message}"
    except Exception as exc:                       # noqa: BLE001
        text, called, sqls, error = "", [], [], f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - started
    expected = set(case["expect_tools"])
    actual = set(called)

    # 答案判定
    wanted = case.get("expect_answer") or []
    norm_answer = normalize(text)
    hits = [w for w in wanted if normalize(w) in norm_answer]
    if not wanted:
        answer_ok = None                            # 没标答案的题不计入答案正确率
    elif case.get("answer_match") == "any":
        answer_ok = bool(hits)
    else:
        answer_ok = len(hits) == len(wanted)

    refusal_ok = None
    if case.get("expect_refusal"):
        refusal_ok = any(m in text for m in REFUSAL_MARKERS)

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected": sorted(expected),
        "called": called,
        "strict": actual == expected,
        "loose": expected <= actual,
        "over": actual > expected,
        "answer_ok": answer_ok,
        "answer_hits": hits,
        "answer_wanted": wanted,
        "refusal_ok": refusal_ok,
        "answer": text,
        "sqls": sqls,
        "error": error,
        "seconds": round(elapsed, 1),
        "why": case.get("why", ""),
    }


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d * 100:5.1f}%)" if d else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    ap.add_argument("--only", default="", help="按 id 或类别过滤")
    ap.add_argument("--workers", type=int, default=4, help="并发数（连接池上限 4）")
    args = ap.parse_args()

    cfg = get_config()
    if not cfg.agent_configured:
        print("AGENT_API_KEY 未配置，路由评测需要真实模型调用", file=sys.stderr)
        return 1

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if args.only in (c["id"], c["category"])]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有匹配的用例", file=sys.stderr)
        return 1

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"route_{tag}", "pw123456", is_admin=False)

    try:
        n_doc, n_tbl = ingest(user)
        print(f"语料：{n_doc} 篇文档 + {n_tbl} 张表 | 用例 {len(cases)} 题 "
              f"| 模型 {cfg.agent_model} | 并发 {args.workers}\n")

        started = time.monotonic()
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(lambda c: run_case(user, c), cases))
        else:
            rows = [run_case(user, c) for c in cases]
        total_seconds = time.monotonic() - started

        # ---------------- 明细 ----------------
        by_cat: dict[str, list] = defaultdict(list)
        for row in rows:
            by_cat[row["category"]].append(row)

        for cat, group in by_cat.items():
            print("=" * 78)
            print(cat)
            print("=" * 78)
            for r in group:
                mark = "✓" if r["strict"] else ("~" if r["loose"] else "✗")
                exp = ",".join(r["expected"]) or "(不调工具)"
                got = ",".join(r["called"]) or "(未调工具)"
                print(f"  {mark} {r['id']:8} 期望[{exp}] 实际[{got}]  {r['seconds']}s")
                if not r["strict"]:
                    print(f"      问题：{r['question']}")
                if r["answer_ok"] is False:
                    missing = [w for w in r["answer_wanted"] if w not in r["answer_hits"]]
                    print(f"      答案未命中 {missing}：{r['answer'][:110]}")
                if r["refusal_ok"] is False:
                    print(f"      ⚠ 该题应承认查不到，但模型给了答案：{r['answer'][:110]}")
                if r["error"]:
                    print(f"      错误：{r['error']}")
            print()

        # ---------------- 汇总 ----------------
        print("=" * 78)
        print("路由准确率（主指标）")
        print("=" * 78)
        print(f"{'类别':<16}{'严格':>16}{'宽松':>16}{'过度调用':>14}")
        for cat, group in by_cat.items():
            n = len(group)
            print(f"{cat:<16}{pct(sum(r['strict'] for r in group), n):>16}"
                  f"{pct(sum(r['loose'] for r in group), n):>16}"
                  f"{pct(sum(r['over'] for r in group), n):>14}")
        n = len(rows)
        print("-" * 78)
        print(f"{'总计':<16}{pct(sum(r['strict'] for r in rows), n):>16}"
              f"{pct(sum(r['loose'] for r in rows), n):>16}"
              f"{pct(sum(r['over'] for r in rows), n):>14}")

        scored = [r for r in rows if r["answer_ok"] is not None]
        refusals = [r for r in rows if r["refusal_ok"] is not None]
        print(f"\n答案正确率（次要指标）：{pct(sum(r['answer_ok'] for r in scored), len(scored))}")
        if refusals:
            print(f"查不到时如实承认：    {pct(sum(r['refusal_ok'] for r in refusals), len(refusals))}")
        print(f"总耗时 {total_seconds:.0f}s，平均每题 {total_seconds / n:.1f}s")

        # ---------------- badcase ----------------
        bad = [r for r in rows if not r["strict"] or r["answer_ok"] is False
               or r["refusal_ok"] is False]
        if bad:
            print(f"\n{'=' * 78}\nbadcase 清单（{len(bad)} 题）\n{'=' * 78}")
            for r in bad:
                reasons = []
                if not r["loose"]:
                    reasons.append("漏调工具")
                elif r["over"]:
                    reasons.append("过度调用")
                if r["answer_ok"] is False:
                    reasons.append("答案错")
                if r["refusal_ok"] is False:
                    reasons.append("该拒答却编了")
                print(f"\n  [{r['id']}] {'/'.join(reasons)}")
                print(f"    问：{r['question']}")
                print(f"    期望 {r['expected'] or '不调工具'} → 实际 {r['called'] or '未调工具'}")
                for sql in r["sqls"]:
                    print(f"    SQL：{sql[:160]}")
                print(f"    答：{r['answer'][:200]}")
                if r["why"]:
                    print(f"    这题为什么在集里：{r['why']}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "routing.json"
        out.write_text(json.dumps({
            "model": cfg.agent_model,
            "embedding_model": cfg.embedding_model,
            "embedding_dimensions": cfg.embedding_dimensions,
            "cases": len(rows),
            "strict": sum(r["strict"] for r in rows),
            "loose": sum(r["loose"] for r in rows),
            "over": sum(r["over"] for r in rows),
            "answer_scored": len(scored),
            "answer_ok": sum(r["answer_ok"] for r in scored),
            "seconds": round(total_seconds),
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入 {out.relative_to(ROOT)}")

        return 0 if all(r["strict"] for r in rows) else 2

    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
