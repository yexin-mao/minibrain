"""幻觉评测：模型说的话有没有出处，该拒答的有没有拒答。

## 两个指标

**溯源率**（自动、确定性）
  把答案里的数字和标识符抽出来，逐个去**两条链路的证据**里找。
  找不到的逐条列出。见 scripts/grounding.py。

  ★ 未溯源 ≠ 幻觉。问「公司总共多少人」答 84，84 不在任何证据里，
  它是 32+14+18+9+11 算出来的，完全正确。所以本脚本按题型分开报，
  聚合类的未溯源数字单独看。

**拒答正确率**（人工标注）
  `eval/probes_hallucination.json` 里 8 道题的正确答案就是"查不到"，
  4 道对照题必须答出来。

  ★ 对照组不是凑数：**把系统改成什么都不答，拒答率就是 100%**。
  没有对照组，一个哑巴系统会显得防幻觉做得最好。

## 为什么不用 LLM 当裁判

RAGAS 的 Faithfulness 要另一个模型来判：不确定性叠加、分数没法调试、进不了 CI。
这里做的是确定性字符串溯源——**不是打一个分，是指出具体哪个数字没出处**。

它抓得住编造的数字和编号，抓不住编造的中文描述。
**是下限不是全貌：报出来的一定可疑，没报的不一定干净。**

跑法：uv run --no-sync python scripts/probe_grounding.py --rounds 3
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from grounding import check                    # noqa: E402
from minibrain import gateway, identity        # noqa: E402
from minibrain.agent.loop import _system_prompt, answer   # noqa: E402
from minibrain.config import get_config        # noqa: E402
from minibrain.contracts import ModuleError    # noqa: E402
from minibrain.db import close_all             # noqa: E402
from minibrain.scripts_purge import purge_user  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_DOC = ROOT / "eval" / "corpus"
CORPUS_TBL = ROOT / "eval" / "corpus_table"
OUT_DIR = ROOT / "eval" / "results"

HALLUCINATION = ROOT / "eval" / "probes_hallucination.json"
MULTIHOP = ROOT / "eval" / "probes_multihop.json"
PROBES = ROOT / "eval" / "probes.json"

# 认定"如实承认查不到"的判据。
#
# ★ 第一版是一张关键词表，产生了**假阴性**：3 例判为"该拒答却答了"的，
#   实际全都拒答了，只是措辞没在表里——
#     「公司没有名为"运维部"的独立部门」   表里有"不存在"，没有"没有名为"
#     「当前知识库中没有关于张敏月薪的信息」 表里有"没有相关"，没有"没有关于"
#
#   这是**指标自己的 bug**，不是模型的问题。而且这是本项目第三次踩到
#   「先测出指标错，再测出系统错」（前两次：mix-06 标注错了笔数、
#   Agent 覆盖率把跨链路多跳误判成失败）。
#
# 修法上刻意不照着失败样本一个个补词——那是在测试集上调参。
# 改成按**语义类别**写正则：否定存在 / 否定可得 / 否定归属。
#
# ★★ 但正则版仍然有假阴性，而且**这是方法本身的上限，不是模式写得不够多**。
#    实测 33 次判定里报 3 次失败，人工核对**全部是措辞没匹配上**：
#      「公司没有设立独立的运维部」    —— 有"没有设立"
#      「郗昭不属于公司员工」          —— 有"不属于…员工"
#      「没有查询到…离职率信息」       —— 有"没有查询到"
#    真实拒答正确率 100%，检测器报 91%。
#
#    继续补模式就是在测试集上调参。所以**到此为止**，并明确记录：
#      本检测器是**下限**——它报的失败要人工核，它报的成功是可信的。
#
#    更深的结论写在 eval/RESULTS.md 探针六：
#      确定性字符串匹配能可靠地测「有没有出处」（数字和标识符是离散的、
#      有限的、可穷举的），但测不好「有没有拒答」（自然语言表达"我不知道"
#      的方式是无穷的）。**这正是 Faithfulness 这类指标要用 LLM 当裁判的原因。**
_REFUSAL_PATTERNS = [
    # 否定存在：没有 X / 不存在 X / 没有名为 X 的 Y
    r"(没有|未|不)(查到|找到|检索到|记录|提及|涉及|包含|提供)",
    r"(没有|不存在)[^，。；\n]{0,12}(信息|记录|数据|说明|字段|列|表|部门|项目|版本|条目)",
    r"没有名为",
    r"(库|文档|资料|语料|知识库|表)(中|里)?(没有|未)",
    # 否定可得：无法 X
    r"无法(获知|得知|确定|回答|提供|查询|判断|推断|预测|计算)",
    # 否定归属：问题预设了错误的身份
    r"不属于(公司)?编制",
    r"并非(公司)?(员工|部门)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS))


def load_cases() -> list[dict]:
    cases = [dict(c, group=f"幻觉·{c['kind']}")
             for c in json.loads(HALLUCINATION.read_text(encoding="utf-8"))]
    # 正常题也一起跑：溯源率要有个基线，不然不知道 90% 算好还是坏
    cases += [dict(c, group="正常·多跳")
              for c in json.loads(MULTIHOP.read_text(encoding="utf-8"))]
    for probe in json.loads(PROBES.read_text(encoding="utf-8")):
        if probe["category"] in ("对照组·单点事实", "全局聚合"):
            cases.append(dict(probe, group=f"正常·{probe['category']}"))
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


def run_case(user, case: dict) -> dict:
    try:
        result = answer(user, case["question"])
        text, evidence = result.answer, result.evidence
        error = None
    except ModuleError as exc:
        text, evidence, error = "", [], f"{exc.code}: {exc.message}"
    except Exception as exc:                                # noqa: BLE001
        text, evidence, error = "", [], f"{type(exc).__name__}: {exc}"

    # ★ "出处"= 模型能看到的一切，三部分都要算：
    #   1. 文档片段（vector_search 的返回）
    #   2. SQL 结果（table_query 的返回）
    #   3. **system prompt**——里面有表名、列名、列取值、文档清单
    #
    # 第 3 条是实测补上的。原来只算 1+2，结果报出一批假阳性：
    #   hal-05 的 "977" 来自表名 t_1ea2c977
    #   hal-04 列举的 PRJ-2026-0178 等真实编号来自文档清单
    # 这些模型确实"看到过"，说出来不是编的。
    #
    # 加进来不会削弱幻觉检测：伪造的 PRJ-2026-0999 既不在证据里，
    # 也不在文档清单里（清单只列真实文件）。
    report = check(text, [e.snippet for e in evidence] + [case["_prompt"]])

    refused = bool(_REFUSAL_RE.search(text))
    expect_refusal = bool(case.get("expect_refusal"))
    refusal_ok = (refused == expect_refusal) if "expect_refusal" in case else None

    wanted = case.get("expect_answer") or []
    norm = re.sub(r"[,\s，、]", "", text)
    if not wanted:
        answer_ok = None
    elif case.get("answer_match") == "any":
        answer_ok = any(re.sub(r"[,\s，、]", "", w) in norm for w in wanted)
    else:
        answer_ok = all(re.sub(r"[,\s，、]", "", w) in norm for w in wanted)

    return {
        "id": case["id"], "group": case["group"], "question": case["question"],
        "answer": text, "error": error,
        "refused": refused, "expect_refusal": expect_refusal, "refusal_ok": refusal_ok,
        "answer_ok": answer_ok, **report,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = get_config()
    if not cfg.agent_configured:
        print("AGENT_API_KEY 未配置", file=sys.stderr)
        return 1

    cases = load_cases()
    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"ground_{tag}", "pw123456", is_admin=False)

    try:
        n_doc, n_tbl = ingest(user)
        print(f"语料 {n_doc} 篇 + {n_tbl} 张表 | 用例 {len(cases)} 题 | {args.rounds} 轮\n")

        prompt = _system_prompt(user)
        for case in cases:
            case["_prompt"] = prompt

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

        print("\n" + "=" * 80)
        print("主表")
        print("=" * 80)
        print(f"  {'分组':<20}{'拒答正确率':>12}{'答案正确率':>12}{'溯源率':>10}{'未溯源条目':>12}")
        for group in sorted(by_group):
            g = by_group[group]
            ref = [r for r in g if r["refusal_ok"] is not None]
            acc = [r for r in g if r["answer_ok"] is not None]
            rates = [r["grounding_rate"] for r in g if r["grounding_rate"] is not None]
            ungrounded = sum(len(r["ungrounded_numbers"]) + len(r["ungrounded_identifiers"])
                             for r in g)
            ref_t = f"{sum(r['refusal_ok'] for r in ref) / len(ref):>11.0%}" if ref else "—".rjust(12)
            acc_t = f"{sum(r['answer_ok'] for r in acc) / len(acc):>11.0%}" if acc else "—".rjust(12)
            gr_t = f"{sum(rates) / len(rates):>9.0%}" if rates else "—".rjust(10)
            print(f"  {group:<20}{ref_t}{acc_t}{gr_t}{ungrounded:>12}")

        print("\n  ★ 对照组不是凑数：把系统改成什么都不答，拒答率就是 100%。")
        print("    「幻觉·对照·应当答出」那组必须保持高答案正确率，否则就是改哑巴了。")

        print("\n" + "=" * 80)
        print("该拒答却答了的（真幻觉）")
        print("=" * 80)
        bad = [r for r in rows if r["expect_refusal"] and not r["refused"]]
        if not bad:
            print("  ✓ 没有")
        for r in bad:
            print(f"\n  [{r['id']} 第{r['round']}轮] {r['question']}")
            print(f"    答：{r['answer'][:150]}")
            if r["ungrounded_numbers"] or r["ungrounded_identifiers"]:
                print(f"    ★ 无出处：数字 {r['ungrounded_numbers']} "
                      f"标识符 {r['ungrounded_identifiers']}")

        print("\n" + "=" * 80)
        print("未溯源条目明细（未溯源 ≠ 幻觉，聚合类的计算结果本来就不在证据里）")
        print("=" * 80)
        for group in sorted(by_group):
            items = [(r["id"], r["ungrounded_numbers"], r["ungrounded_identifiers"])
                     for r in by_group[group]
                     if r["ungrounded_numbers"] or r["ungrounded_identifiers"]]
            if items:
                print(f"\n  ── {group} ──")
                for pid, nums, ids in items[:6]:
                    print(f"    {pid:<9}数字 {nums}  标识符 {ids}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "probe_grounding.json"
        out.write_text(json.dumps({
            "model": cfg.agent_model, "rounds": args.rounds, "cases": len(cases),
            "rows": [{k: v for k, v in r.items() if k != "_prompt"} for r in rows],
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
