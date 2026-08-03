"""对照实验：多跳失败到底是「召回率不够」还是「召回时机不对」？

这个区分决定了该往哪个方向修：
  - 如果是召回率问题 → 调 k、换模型、加 rerank 就能解决，不需要换范式
  - 如果是时机问题   → 单次检索天然做不到，必须改变知识组织方式（GraphRAG）
                       或改变检索过程（Agentic RAG / 迭代检索）

做法：同一道题，同样的 k，对比单次检索 vs 两步检索。

跑法：uv run --no-sync python scripts/probe_multihop.py
"""

from __future__ import annotations

import pathlib
import sys
import uuid

sys.path.insert(0, "src")

from minibrain import gateway, identity          # noqa: E402
from minibrain.db import close_all               # noqa: E402
from minibrain.scripts_purge import purge_user   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
K = 3
TARGET = "dept-tech.md"          # 答对本题必须召回的那一篇
QUESTION = "张敏的上级的上级是谁？"


def files_of(user, query: str, k: int) -> list[str]:
    result = gateway.search("vector-rag", user, query, top_k=k)
    seen, out = set(), []
    for e in result.evidence:
        name = e.location.split(" #")[0]
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def main() -> int:
    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"hop_{tag}", "pw123456", is_admin=False)

    try:
        for path in sorted(CORPUS.glob("*.md")):
            doc_id = gateway.call(
                "vector-rag", "upload_document", user, None, path.name,
                path.read_text(encoding="utf-8"),
            )
            gateway.process("vector-rag", doc_id)

        print(f"题目：{QUESTION}")
        print(f"必须召回：{TARGET}（技术部→李伟）")
        print(f"两种方式都用同样的 top_k = {K}\n")

        # ---------- 方式一：单次检索 ----------
        print("─" * 68)
        print("方式一：拿原问题直接检索一次")
        print("─" * 68)
        one = files_of(user, QUESTION, K)
        for i, f in enumerate(one, 1):
            print(f"  {i}. {f}")
        print(f"\n  结果：{'✓ 命中' if TARGET in one else '✗ 漏掉 ' + TARGET}")

        # ---------- 方式二：两步检索 ----------
        print("\n" + "─" * 68)
        print("方式二：先查实体，读到桥接事实后再发起第二次检索")
        print("─" * 68)

        print(f'  第 1 步  查「张敏」')
        step1 = files_of(user, "张敏", K)
        for i, f in enumerate(step1, 1):
            print(f"           {i}. {f}")

        # 读完第一跳才知道下一步该查什么 —— 这正是单次检索缺的那个环节
        bridge = "后端组"
        print(f'\n           从 team-backend.md 读到桥接事实：张敏 → {bridge}')

        print(f'\n  第 2 步  查「{bridge}属于哪个部门，负责人是谁」')
        step2 = files_of(user, f"{bridge}属于哪个部门，负责人是谁", K)
        for i, f in enumerate(step2, 1):
            print(f"           {i}. {f}")

        two = set(step1) | set(step2)
        print(f"\n  结果：{'✓ 命中' if TARGET in two else '✗ 仍然漏掉'}")

        # ---------- 结论 ----------
        print("\n" + "=" * 68)
        if TARGET not in one and TARGET in two:
            print("结论：同样的 k，单次检索失败，两步检索成功。")
            print("      信息一直都在库里，缺的不是召回能力，是「读完第一跳再决定下一步查什么」")
            print("      这个环节。单次检索在架构上就没有这个环节。")
            print()
            print("      两条修法：")
            print("        改变知识组织 → 把关系预先抽成图，多跳变成图遍历（GraphRAG）")
            print("        改变检索过程 → 让 Agent 迭代检索，自己决定下一跳（Agentic RAG）")
        elif TARGET in one:
            print("结论：单次检索本次就命中了。换个多跳题目再测，或调小 k。")
        else:
            print("结论：两步也没命中。检查语料或 embedding 配置。")
        print("=" * 68)
        return 0

    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
