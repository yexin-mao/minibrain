"""轨迹评测：不看答案对不对，看**它是怎么走到答案的**。

## 为什么需要这个

现有的所有评测衡量的都是**终点**：答案对不对、召回全不全。
那是 RAG 岗的答案。Agent 岗真正会追问的是**路径**：

    「你怎么知道你的 agent 是对的？」
    「多查的那一轮，到底带来新东西了吗，还是把同一个问题换个说法又查了一遍？」
    「它什么时候会打转？打转了你能发现吗？」

`agent/loop.py` 里 `trace: list[ToolCallTrace]` **早就在记录每次工具调用的
名字、参数、结果**——但从来没有人算过它。这个脚本就是来算的。

★ 已有的 `probe_agentic.py` 只存了 `calls = [t.name for t in trace]`，
  丢掉了参数和每轮的证据。所以最有价值的两个指标它算不出来，必须重跑。

## 四个指标，以及各自能回答哪道面试题

### 1. 边际证据收益（marginal evidence gain）★ 最值钱

第 n 轮检索带来了多少条**前面没出现过**的证据。

- 第 2、3 轮的边际收益接近 0 → **agentic 在烧钱不干活**，该收紧停止条件
- 边际收益高 → 这就是「为什么要做 Agentic RAG」的直接数据

一句话能回答：「你为什么上多轮检索，凭什么说它值？」

### 2. 冗余检索率

同一轮次里语义上重复的查询（这里用**归一化后完全相同**这个保守判据——
宁可少报，不可虚报。真要做语义去重得引入相似度阈值，那本身又是个可调参数，
会把「测量」变成「又一个要调的东西」）。

回答：「你的 agent 会不会原地打转？」

### 3. 轮数分布

不只报平均值，报**分布和长尾**。平均 2.1 轮和「大部分 1 轮、少数撞满上限」
是完全不同的系统，而平均值把这个区别抹平了。

回答：「一次问答烧几轮？最坏情况呢？」

### 4. 工具调用失败率

工具报错、返回空、参数非法各占多少，以及**报错之后模型做了什么**——
改参数重试、换工具、还是继续撞墙。

回答：「工具失败了你的 agent 会怎样？」

## 一条刻意的取舍

本脚本**不**评价答案对错——那是 `probe_agentic.py` 和 grounding 评测的事。
两件事分开量，混在一起会得到「答案对了所以路径也对」这种错误推论：
一个瞎查五轮最后蒙对的 agent，和一个两轮直达的 agent，答案分数一样，
**但它们不是一个东西**。

跑法：uv run --no-sync python scripts/eval_trajectory.py
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys
import uuid

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from minibrain import gateway, identity                           # noqa: E402
from minibrain.handwritten.agent_loop import answer                           # noqa: E402
from minibrain.config import get_config                           # noqa: E402
from minibrain.contracts import ModuleError                       # noqa: E402
from minibrain.db import close_all                                # noqa: E402
from minibrain.scripts_purge import purge_user                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
OUT_DIR = ROOT / "eval" / "results"


def load_cases(limit: int | None) -> list[dict]:
    """直接复用 probe_agentic.py 的用例组装。

    ★ 刻意不另起一套题。同一批题才能和历史的 agentic 结果对上——
      换一批题就没法说「轨迹变好了」还是「题变简单了」。
      probe_agentic 那套里有对照组（单点事实），
      它是用来证明「续查只在该触发时触发」的：没有对照组，
      一个每题都查三遍的实现也会显得很成功。
    """
    from probe_agentic import load_cases as _load          # noqa: PLC0415
    cases = _load()
    return cases[:limit] if limit else cases


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


def _query_of(call) -> str:
    """从工具调用参数里取出「这次到底查了什么」，归一化后用于冗余判断。

    两种工具的「查询」长得不一样：vector_search 是自然语言 query，
    table_query 是一条 SQL。都归一化成小写、压掉空白。
    """
    try:
        args = json.loads(call.arguments) if call.arguments else {}
    except json.JSONDecodeError:
        return ""
    raw = args.get("query") or args.get("sql") or ""
    return re.sub(r"\s+", " ", str(raw).strip().lower())


def _evidence_keys(preview: str) -> set[str]:
    """从工具返回的文本里抠出证据标识（文件名 / 表名）。

    ★ 这里用的是 result_preview（截断到 300 字），所以**只能看到前几条证据**。
      于是边际收益是**低估**的：后面被截掉的证据不计入。
      低估比高估安全——如果连低估都显示第 2 轮有收益，那就是真的有。
      这条限制必须写进报告，不能让读者以为是精确值。
    """
    return set(re.findall(r"[\w\-.]+\.md", preview)) | set(
        re.findall(r"\bt_[0-9a-f]+\b", preview))


def run_case(user, case: dict) -> dict:
    try:
        result = answer(user, case["question"])
        trace, error = result.trace, None
    except ModuleError as exc:
        return {**case, "error": f"{exc.code}: {exc.message}", "steps": 0,
                "calls": [], "marginal": [], "redundant": 0, "tool_errors": 0}
    except Exception as exc:                                      # noqa: BLE001
        return {**case, "error": f"{type(exc).__name__}: {exc}", "steps": 0,
                "calls": [], "marginal": [], "redundant": 0, "tool_errors": 0}

    seen_evidence: set[str] = set()
    seen_queries: set[str] = set()
    marginal, redundant, tool_errors = [], 0, 0
    calls = []

    for call in trace:
        query = _query_of(call)
        if query and query in seen_queries:
            redundant += 1
        seen_queries.add(query)

        keys = _evidence_keys(call.result_preview)
        new = keys - seen_evidence
        marginal.append(len(new))
        seen_evidence |= keys

        # 工具层面的失败：报错、没查到、参数不合法
        preview = call.result_preview
        failed = any(w in preview for w in
                     ("错误", "失败", "不合法", "没有查到", "没有命中", "为空"))
        tool_errors += failed

        calls.append({"name": call.name, "query": query[:80],
                      "new_evidence": len(new), "failed": failed})

    return {**case, "error": error, "steps": len(trace), "calls": calls,
            "marginal": marginal, "redundant": redundant,
            "tool_errors": tool_errors,
            "total_evidence": len(seen_evidence)}


def summarize(rows: list[dict]) -> dict:
    """全局汇总 + **按组拆开**。

    ★★ 按组拆开不是锦上添花，是这个评测能不能成立的关键。

    用例里有对照组（单点事实类），它们**本来就该一轮结束**。
    如果只报全局平均，对照组的 1 步会把多跳题的 2 步稀释掉，
    「边际收益」也会被一堆「没有第 2 轮」的用例拉平。

    更糟的是反过来：**没有对照组的话，一个每题都查三遍的实现看起来会很成功**
    ——它在多跳题上表现完美，而你看不见它在简单题上浪费了多少。
    对照组存在的意义就是证明「续查只在该触发时触发」，
    而这一点只有分组看才看得出来。
    """
    ok = [r for r in rows if not r["error"]]
    steps = [r["steps"] for r in ok]

    groups: dict[str, dict] = {}
    for name in sorted({r.get("group", "未分组") for r in ok}):
        members = [r for r in ok if r.get("group", "未分组") == name]
        g_steps = [r["steps"] for r in members]
        g_round: dict[int, list[int]] = collections.defaultdict(list)
        for r in members:
            for i, gain in enumerate(r["marginal"], start=1):
                g_round[i].append(gain)
        groups[name] = {
            "cases": len(members),
            "steps_mean": statistics.mean(g_steps),
            "steps_hist": dict(sorted(collections.Counter(g_steps).items())),
            # 「续查率」= 走了 2 步以上的比例。对照组这个数应该接近 0，
            # 多跳组应该接近 1。两边都对，才说明停止判定是「按需触发」而不是「一律多查」。
            "follow_up_rate": sum(1 for s in g_steps if s >= 2) / len(g_steps),
            "marginal_by_round": {
                str(k): round(statistics.mean(v), 2) for k, v in sorted(g_round.items())},
            "redundant": sum(r["redundant"] for r in members),
        }

    # 边际收益按轮次对齐：第 1 轮 / 第 2 轮 / 第 3 轮…各自的平均新增证据数
    by_round: dict[int, list[int]] = collections.defaultdict(list)
    for r in ok:
        for i, gain in enumerate(r["marginal"], start=1):
            by_round[i].append(gain)

    return {
        "cases": len(rows),
        "failed_cases": len(rows) - len(ok),
        "steps_mean": statistics.mean(steps) if steps else 0,
        "steps_median": statistics.median(steps) if steps else 0,
        "steps_max": max(steps) if steps else 0,
        "steps_hist": dict(sorted(collections.Counter(steps).items())),
        "hit_step_limit": sum(1 for s in steps if s >= get_config().agent_max_steps),
        "redundant_calls": sum(r["redundant"] for r in ok),
        "tool_errors": sum(r["tool_errors"] for r in ok),
        "marginal_by_round": {
            str(k): {"n": len(v), "mean_new_evidence": statistics.mean(v)}
            for k, v in sorted(by_round.items())},
        "by_group": groups,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个用例")
    ap.add_argument("--label", default="current")
    args = ap.parse_args()

    cases = load_cases(args.limit)
    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"traj_{tag}", "pw123456", is_admin=False)

    try:
        total = ingest(user)
        print(f"语料 {total} 篇 | 用例 {len(cases)} 个 | "
              f"max_steps={get_config().agent_max_steps}\n")

        rows = []
        for i, case in enumerate(cases, start=1):
            row = run_case(user, case)
            rows.append(row)
            mark = "✗" if row["error"] else " "
            print(f"  [{i:>2}/{len(cases)}]{mark} {row['id']:<8} "
                  f"{row['steps']} 步  新增证据 {row['marginal']}")

        s = summarize(rows)

        print("\n" + "=" * 78)
        print("轨迹评测：不看答案对不对，看它是怎么走到答案的")
        print("=" * 78)

        print(f"\n  轮数分布（平均 {s['steps_mean']:.2f} / 中位数 {s['steps_median']:.0f} "
              f"/ 最大 {s['steps_max']}）：")
        for k, v in s["steps_hist"].items():
            print(f"    {k} 步：{'█' * v} {v}")
        print(f"    撞上 max_steps 上限的：{s['hit_step_limit']} 个")
        print("    ★ 只看平均值会骗人：平均 2 轮可能是「都 2 轮」，"
              "也可能是「大半 1 轮 + 少数撞满」")

        print("\n  ★ 边际证据收益 —— 第 n 轮带来了多少条前面没见过的证据：")
        for k, v in s["marginal_by_round"].items():
            print(f"    第 {k} 轮：{v['mean_new_evidence']:>5.2f} 条新证据"
                  f"（{v['n']} 次调用）")
        print("    ★ 这一列如果第 2 轮就接近 0，说明多轮检索在烧钱不干活。")
        print("    ⚠ 是**低估值**：证据从 result_preview（截断 300 字）里抠，"
              "后面的看不到。")

        print("\n  分组拆开（★ 对照组是用来证明「续查只在该触发时触发」的）：")
        print(f"    {'组':<12}{'用例':>5}{'平均步数':>10}{'续查率':>9}   各轮新增证据")
        for name, g in s["by_group"].items():
            rounds = " / ".join(f"第{k}轮 {v}" for k, v in g["marginal_by_round"].items())
            print(f"    {name:<12}{g['cases']:>5}{g['steps_mean']:>10.2f}"
                  f"{g['follow_up_rate']:>9.0%}   {rounds}")
        print("    ★ 对照组续查率应接近 0、多跳组应接近 1。"
              "两边都对才说明是「按需触发」而不是「一律多查」。")

        print(f"\n  冗余检索（查询字面完全重复）：{s['redundant_calls']} 次")
        print(f"  工具层面失败：{s['tool_errors']} 次")
        print(f"  整体失败用例：{s['failed_cases']} / {s['cases']}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"trajectory_{args.label}.json"
        out.write_text(json.dumps(
            {"label": args.label, "corpus_files": total,
             "model": get_config().agent_model,
             "max_steps": get_config().agent_max_steps,
             "summary": s, "rows": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细已写入 {out.relative_to(ROOT)}")
        return 0
    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
