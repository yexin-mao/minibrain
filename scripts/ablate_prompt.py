"""消融实验：system prompt 里加什么，才能让 Agent 选对链路？

背景：43 题路由评测测出「文档·组织事实」类只有 1/6，三轮一致。
根因诊断是 system prompt 里的信息不对等——表格侧列了表名/列名/行数，
文档侧只有一句"适合非结构化文档"。模型看得见表有哪些列，
却完全不知道文档里写了什么，于是两边都有的事实一律去查表。

诊断只是假设。这个脚本用**消融实验**验证它：逐段往 prompt 里加东西，
每次只加一段，其余一字不改，看命中率怎么变。

四个版本，层层叠加：

  A  现状（基线）
  B  A + 光加文件名清单
  C  A + 文件名 + 文档标题
  D  C + 权威来源规则（同一事实两边都有时以谁为准）

为什么要分这么细：如果一口气把 B/C/D 全加上去然后变好了，
你只知道"加了一堆东西有效"，不知道**哪一段起了作用**。
面试问"你为什么这么改"，答不出根因就只是碰运气。

—— 便宜的代理指标 ——
本脚本只看**第一个被调用的工具**，不跑完整 tool loop、不执行工具。
一题一次模型调用，43 题 × 4 版本 × N 轮很快跑完，适合筛方案。
选定方案后必须用 scripts/eval_routing.py 做完整验证
（它跑真实 Agent、看全部工具调用集合、算严格/宽松/过度调用三个指标）。

  筛方案 → 本脚本（便宜、可比多个版本）
  下结论 → eval_routing.py（完整、和基线同口径）

跑法：
  uv run --no-sync python scripts/ablate_prompt.py
  uv run --no-sync python scripts/ablate_prompt.py --rounds 1     # 快速看一眼
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")

from openai import OpenAI                          # noqa: E402

from minibrain import gateway, identity            # noqa: E402
from minibrain.agent.loop import _system_prompt    # noqa: E402
from minibrain.agent.tools import TOOL_SCHEMAS     # noqa: E402
from minibrain.config import get_config            # noqa: E402
from minibrain.db import close_all                 # noqa: E402
from minibrain.scripts_purge import purge_user     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_DOC = ROOT / "eval" / "corpus"
CORPUS_TBL = ROOT / "eval" / "corpus_table"
GOLDEN = ROOT / "eval" / "routing.json"
OUT_DIR = ROOT / "eval" / "results"

# 插入锚点：所有新增内容都插在表格清单**之前**，这样每个版本只多一段，其余完全一致。
ANCHOR = "可查询的数据表（只列出你有权访问的）："

AUTHORITY_RULES = """权威来源规则（重要）：同一个事实可能在文档和表格里都出现，此时以下面的规定为准：
- 组织架构、岗位职责、制度规定、项目信息 → **以文档为准**，用 vector_search
- 人数、金额、日期等需要统计计算的 → 以表格为准，用 table_query
- 表格里的行可能包含记账用的辅助行，不代表真实的组织单元。
"""


def build_variants(user) -> dict[str, str]:
    """四个版本，层层叠加，除新增段落外一字不改。"""
    base = _system_prompt(user)
    names_only = gateway.call("vector-rag", "describe_corpus", user, with_titles=False)
    with_titles = gateway.call("vector-rag", "describe_corpus", user, with_titles=True)

    def insert(block: str) -> str:
        return base.replace(ANCHOR, block + "\n" + ANCHOR, 1)

    catalog_bare = f"可检索的文档（只列出你有权访问的）：\n{names_only}\n"
    catalog_full = f"可检索的文档（只列出你有权访问的）：\n{with_titles}\n"

    return {
        "A 现状（基线）": base,
        "B +文件名": insert(catalog_bare),
        "C +文件名和标题": insert(catalog_full),
        "D +权威来源规则": insert(catalog_full + "\n" + AUTHORITY_RULES),
    }


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


def first_tool(client, model: str, system_text: str, question: str) -> str | None:
    """只问一次，只看它**第一个**想调的工具。不执行工具，不进循环。"""
    try:
        response = client.chat.completions.create(
            model=model, temperature=0, tools=TOOL_SCHEMAS,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": question},
            ],
        )
    except Exception:                              # noqa: BLE001
        return "(调用失败)"
    calls = response.choices[0].message.tool_calls
    return calls[0].function.name if calls else None


def scores(case: dict, picked: str | None) -> bool:
    """首选命中：第一个调的工具在期望集合里；期望为空时要求一个都没调。

    这是代理指标——期望两个工具的题，只要第一个对就算命中，
    因为本脚本不跑完整循环，看不到它后面还会不会调第二个。
    """
    expected = set(case["expect_tools"])
    if not expected:
        return picked is None
    return picked in expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = get_config()
    if not cfg.agent_configured:
        print("AGENT_API_KEY 未配置，消融实验需要真实模型调用", file=sys.stderr)
        return 1

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    client = OpenAI(base_url=cfg.agent_base_url, api_key=cfg.agent_api_key)

    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"ablate_{tag}", "pw123456", is_admin=False)

    try:
        n_doc, n_tbl = ingest(user)
        variants = build_variants(user)

        print(f"语料 {n_doc} 篇文档 + {n_tbl} 张表 | 用例 {len(cases)} 题 "
              f"| {len(variants)} 个版本 × {args.rounds} 轮 | 模型 {cfg.agent_model}")
        print("指标：首选命中率（只看第一个调用的工具，代理指标）\n")

        print("各版本 system prompt 长度：")
        for name, text in variants.items():
            print(f"  {name:<18}{len(text):>6} 字符")
        print()

        results: dict[str, list[dict]] = defaultdict(list)

        for round_no in range(1, args.rounds + 1):
            for name, system_text in variants.items():
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    picks = list(pool.map(
                        lambda c: first_tool(client, cfg.agent_model, system_text, c["question"]),
                        cases,
                    ))
                for case, picked in zip(cases, picks):
                    results[name].append({
                        "round": round_no, "id": case["id"], "category": case["category"],
                        "expected": sorted(case["expect_tools"]), "picked": picked,
                        "ok": scores(case, picked),
                    })
            print(f"  第 {round_no} 轮完成")

        # ---------------- 总分 ----------------
        print("\n" + "=" * 78)
        print("首选命中率（全部题目，跨轮合计）")
        print("=" * 78)
        for name in variants:
            rows = results[name]
            ok = sum(r["ok"] for r in rows)
            per_round = [
                sum(r["ok"] for r in rows if r["round"] == i)
                for i in range(1, args.rounds + 1)
            ]
            detail = " / ".join(f"{v}" for v in per_round)
            print(f"  {name:<18}{ok:>4}/{len(rows):<4}({ok / len(rows) * 100:5.1f}%)   "
                  f"逐轮：{detail}（每轮满分 {len(cases)}）")

        # ---------------- 分类别 ----------------
        categories = sorted({c["category"] for c in cases})
        print("\n" + "=" * 78)
        print("分类别首选命中率")
        print("=" * 78)
        header = "".join(f"{n.split()[0]:>10}" for n in variants)
        print(f"  {'类别':<18}{header}")
        for cat in categories:
            cells = ""
            for name in variants:
                rows = [r for r in results[name] if r["category"] == cat]
                cells += f"{sum(r['ok'] for r in rows) / len(rows) * 100:>9.0f}%"
            print(f"  {cat:<18}{cells}")

        # ---------------- 回归检查 ----------------
        print("\n" + "=" * 78)
        print("回归检查：原本就对的题，有没有被改坏？")
        print("=" * 78)
        base_name = next(iter(variants))
        base_ok = {r["id"] for r in results[base_name] if r["ok"]}
        for name in list(variants)[1:]:
            broke = sorted({
                r["id"] for r in results[name]
                if r["id"] in base_ok and not r["ok"]
            })
            fixed = sorted({
                r["id"] for r in results[name]
                if r["id"] not in base_ok and r["ok"]
            })
            print(f"  {name:<18}修好 {len(fixed):>2} 题   改坏 {len(broke):>2} 题"
                  + (f"  ← {', '.join(broke)}" if broke else ""))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "prompt_ablation.json"
        out.write_text(json.dumps({
            "model": cfg.agent_model,
            "rounds": args.rounds,
            "cases": len(cases),
            "metric": "first_tool_hit_rate",
            "variants": {
                name: {
                    "prompt_chars": len(text),
                    "hit_rate": sum(r["ok"] for r in results[name]) / len(results[name]),
                }
                for name, text in variants.items()
            },
            "rows": {name: results[name] for name in variants},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入 {out.relative_to(ROOT)}")
        print("\n注意：这是代理指标。选定版本后必须用 scripts/eval_routing.py")
        print("      跑完整 Agent 循环，才能和 83.7% 的基线同口径对比。")
        return 0

    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
