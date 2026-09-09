"""Agentic RAG 评测：Agent 会不会自己"再查一次"。

## 为什么需要这个脚本

之前所有检索评测都是**直接调 search()**，一次检索一个排名。
但多跳问题的症结不在单次检索的质量，在于**要不要检索第二次**：

    多跳题（84 篇语料 + 混合检索）
      MRR = 1.000              第一跳文档永远排第 1
      Complete Recall@3 = 0.00 第二跳文档排 7~14 名，从来凑不齐

    张敏的上级的上级是谁？    必需文档排名 [8, 1]
    Alpha 项目的对接人向谁汇报？必需文档排名 [7, 1, 14]

第二跳那篇文档里根本没有问题中的实体（dept-tech.md 里没有"张敏"），
所以**任何改善排序的办法都无效**——rerank、混合检索、换模型都试过或推理过了。
唯一的修法是分两次查：先查到"张敏 → 后端组"，再拿"后端组"去查。

所以要在 **Agent 层**而不是检索层测量：跑真实的 tool loop，
看它累积检索到的文档能不能凑齐。

## 指标

  续查触发率        一道题里 vector_search 被调用 ≥2 次的比例
  累积覆盖率        所有轮次检索到的文档**并集**是否覆盖全部必需文档
                   （这是 Complete Recall 在 Agent 层的对应物）
  答案正确率        预标答案出现在最终回答里
  平均工具调用次数   成本。多轮一定更贵，要量出来贵多少

★ 单点事实题的续查触发率必须保持接近 0——
  不该触发的乱触发，就是白白多花钱多等几秒。这和"关键词路只在有标识符时
  才发声"是同一类问题：能力要有，但要在对的时候才用。

跑法：uv run --no-sync python scripts/probe_agentic.py
      uv run --no-sync python scripts/probe_agentic.py --rounds 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
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
OUT_DIR = ROOT / "eval" / "results"

MULTIHOP = ROOT / "eval" / "probes_multihop.json"
STOP = ROOT / "eval" / "probes_stop.json"
PROBES = ROOT / "eval" / "probes.json"


def load_cases() -> list[dict]:
    """多跳题（新增 15 道）+ 原有多跳/全局聚合 + 对照组（单点事实）。

    对照组不是凑数：它是用来证明"续查只在该触发时触发"的。
    没有对照组，一个每题都查三遍的实现也会显得很成功。
    """
    cases = [dict(c, group="多跳·新增") for c in json.loads(MULTIHOP.read_text(encoding="utf-8"))]
    # 压力题：专门构造来触发"查到耗尽"。靠跑很多轮等罕见事件太慢，
    # 直接造出容易跑飞的场景（答案不存在 / 链条到顶 / 三跳以上 / 指代不清 / 前提错误）。
    if STOP.is_file():
        cases += [dict(c, group=f"压力·{c['kind']}", required=[], expect_answer=[])
                  for c in json.loads(STOP.read_text(encoding="utf-8"))]
    group_of = {
        "多跳推理": "多跳·原有",
        "全局聚合": "全局聚合",
        "对照组·单点事实": "对照·单点事实",
    }
    for probe in json.loads(PROBES.read_text(encoding="utf-8")):
        group = group_of.get(probe["category"])
        if group:
            cases.append(dict(probe, group=group, expect_answer=probe.get("expect_answer", [])))
    return cases


def ingest(user) -> tuple[int, int]:
    docs = sorted(CORPUS_DOC.glob("*.md"))
    for path in docs:
        gateway.process("vector-rag", gateway.call(
            "vector-rag", "upload_document", user, None,
            path.name, path.read_text(encoding="utf-8")))
    tables = sorted(CORPUS_TBL.glob("*.csv"))
    for path in tables:
        gateway.process("table-rag", gateway.call(
            "table-rag", "upload_csv", user, None, path.name, path.read_bytes()))
    return len(docs), len(tables)


def normalize(text: str) -> str:
    return re.sub(r"[,\s，、]", "", text)


def run_case(user, case: dict) -> dict:
    try:
        result = answer(user, case["question"])
        text, trace = result.answer, result.trace
        evidence = result.evidence
        error = None
    except ModuleError as exc:
        text, trace, evidence, error = "", [], [], f"{exc.code}: {exc.message}"
    except Exception as exc:                       # noqa: BLE001
        text, trace, evidence, error = "", [], [], f"{type(exc).__name__}: {exc}"

    calls = [t.name for t in trace]
    searches = calls.count("vector_search")

    # 文档链覆盖：所有轮次里 vector_search 检索到的文档并集，是否覆盖必需文档。
    #
    # ★ 这个指标会**低估**系统，必须和答案正确率一起看。
    # 因为花名册 CSV 和文档有意重叠（部门/小组/人数/组长两边都有），
    # Agent 经常走"文档 → 表格"的跨链路多跳：
    #     vector_search（项目文档 → 得知"会计组"）→ table_query（花名册 → 财务部）
    # 那是**更好的**策略——花名册才是人数和归属的权威来源。
    # 但这条路径下 required 文档没被检索，覆盖率判 0。
    #
    # 所以真正的成败判据是 answer_ok；covered 只用来看"文档链自己够不够用"。
    retrieved = {e.location.split(" #")[0] for e in evidence if e.module == "vector-rag"}
    required = set(case.get("required", []))
    covered = bool(required) and required <= retrieved
    used_table = "table_query" in calls

    exhausted = len(calls) >= get_config().agent_max_steps
    # 「给出了结论」：不是超轮兜底那句话。答对、答错、或如实说查不到，都算给出了结论。
    gave_up_cleanly = bool(text) and "轮数已达上限" not in text and "超过最大工具调用轮数" not in text

    wanted = case.get("expect_answer") or []
    norm = normalize(text)
    answer_ok = all(normalize(w) in norm for w in wanted) if wanted else None

    return {
        "id": case["id"], "group": case["group"], "question": case["question"],
        "calls": calls, "searches": searches, "followed_up": searches >= 2,
        "required": sorted(required), "retrieved": sorted(retrieved),
        "missing": sorted(required - retrieved), "covered": covered,
        "used_table": used_table,
        "exhausted": exhausted, "gave_up_cleanly": gave_up_cleanly,
        "answer_ok": answer_ok, "answer": text, "error": error,
        "bridge": case.get("bridge", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default="", help="给这次结果起个名字，便于前后对比")
    args = ap.parse_args()

    cfg = get_config()
    if not cfg.agent_configured:
        print("AGENT_API_KEY 未配置", file=sys.stderr)
        return 1

    cases = load_cases()
    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"agentic_{tag}", "pw123456", is_admin=False)

    try:
        n_doc, n_tbl = ingest(user)
        print(f"语料 {n_doc} 篇 + {n_tbl} 张表 | 用例 {len(cases)} 题 "
              f"| {args.rounds} 轮 | 模型 {cfg.agent_model}\n")

        rows: list[dict] = []
        for round_no in range(1, args.rounds + 1):
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                batch = list(pool.map(lambda c: run_case(user, c), cases))
            for row in batch:
                row["round"] = round_no
            rows.extend(batch)
            print(f"  第 {round_no} 轮完成")

        by_group: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_group[row["group"]].append(row)

        print("\n" + "=" * 78)
        print("主表")
        print("=" * 78)
        print(f"  {'分组':<16}{'续查触发率':>12}{'文档链覆盖':>12}{'答案正确率':>12}{'平均调用次数':>14}{'耗尽率':>9}{'有结论':>9}")
        for group, group_rows in by_group.items():
            n = len(group_rows)
            follow = sum(r["followed_up"] for r in group_rows) / n
            cover = sum(r["covered"] for r in group_rows) / n
            scored = [r for r in group_rows if r["answer_ok"] is not None]
            acc = (sum(r["answer_ok"] for r in scored) / len(scored)) if scored else float("nan")
            avg = sum(len(r["calls"]) for r in group_rows) / n
            acc_text = "—" if scored == [] else f"{acc:>11.1%}"
            exh = sum(r["exhausted"] for r in group_rows) / n
            clean = sum(r["gave_up_cleanly"] for r in group_rows) / n
            print(f"  {group:<16}{follow:>11.1%}{cover:>12.1%}{acc_text:>12}{avg:>14.2f}"
                  f"{exh:>9.1%}{clean:>9.0%}")

        print("\n  ★ 对照组的续查触发率应当接近 0——不该触发的乱触发就是白花钱")
        print("  ★ 文档链覆盖率会低估系统：Agent 常走「文档 → 表格」的跨链路多跳，")
        print("    那条路径下必需文档没被检索，但答案是对的。以答案正确率为准。")

        print("\n" + "=" * 78)
        print("逐题（✓/✗ 是答案正确性；路径列显示它怎么查的）")
        print("=" * 78)
        stop_groups = sorted(g for g in by_group if g.startswith("压力·"))
        if stop_groups:
            print("\n" + "=" * 78)
            print("压力题：会不会停")
            print("=" * 78)
            for group in stop_groups:
                for row in by_group[group]:
                    mark = "✗耗尽" if row["exhausted"] else ("✓" if row["gave_up_cleanly"] else "?")
                    print(f"  {mark:<6}{row['id']:<9}{len(row['calls'])}次  {row['question'][:32]}")
                    if row["exhausted"]:
                        print(f"          调用序列：{row['calls']}")

        for group in ("多跳·新增", "多跳·原有", "全局聚合"):
            if group not in by_group:
                continue
            print(f"\n  ── {group} ──")
            for row in by_group[group]:
                ok = row["answer_ok"]
                mark = "✓" if ok else ("✗" if ok is False else "?")
                path = (f"文档×{row['searches']}" if row["searches"] >= 2
                        else ("文档→表格" if row["used_table"] and row["searches"] else
                              ("纯表格" if row["used_table"] else "文档×1")))
                print(f"  {mark} {row['id']:<8}{path:<11}{row['question'][:30]}")
                if ok is False:
                    print(f"       答：{row['answer'][:64]}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"probe_agentic{'_' + args.label if args.label else ''}.json"
        out = OUT_DIR / name
        out.write_text(json.dumps({
            "label": args.label, "model": cfg.agent_model,
            "corpus_files": n_doc, "rounds": args.rounds, "cases": len(cases),
            "summary": {
                group: {
                    "follow_up_rate": sum(r["followed_up"] for r in g) / len(g),
                    "coverage": sum(r["covered"] for r in g) / len(g),
                    "avg_calls": sum(len(r["calls"]) for r in g) / len(g),
                }
                for group, g in by_group.items()
            },
            "rows": rows,
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
