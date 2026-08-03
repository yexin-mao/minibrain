"""探针：召回不全时，模型会说「我不确定」还是自信地给个错答案？

这是 RAG 在企业场景最危险的失效模式 —— 不是答不出来，是**答错了但看起来对**。
用户拿到一个带来源、有理有据的数字，无从察觉它少算了一个部门。

跑法：uv run --no-sync python scripts/probe_silent_error.py
"""

from __future__ import annotations

import pathlib
import sys
import uuid

sys.path.insert(0, "src")

from minibrain import gateway, identity          # noqa: E402
from minibrain.config import get_config          # noqa: E402
from minibrain.db import close_all               # noqa: E402
from minibrain.scripts_purge import purge_user   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus"
QUESTION = "公司总共大约多少人？"
TRUTH = {"技术部": 32, "产品部": 14, "市场部": 18, "人力资源部": 9, "财务部": 11}


def main() -> int:
    from openai import OpenAI

    cfg = get_config()
    tag = uuid.uuid4().hex[:6]
    user = identity.create_user(f"silent_{tag}", "pw123456", is_admin=False)

    try:
        for path in sorted(CORPUS.glob("*.md")):
            doc_id = gateway.call(
                "vector-rag", "upload_document", user, None, path.name,
                path.read_text(encoding="utf-8"),
            )
            gateway.process("vector-rag", doc_id)

        result = gateway.search("vector-rag", user, QUESTION, top_k=5)
        files = [e.location.split(" #")[0] for e in result.evidence]

        print(f"问题：{QUESTION}")
        print(f"真实答案：{' + '.join(f'{k}{v}' for k, v in TRUTH.items())} = {sum(TRUTH.values())} 人\n")
        print(f"top_k=5 召回：{', '.join(files)}")
        missing = [f for f in
                   ["dept-tech.md", "dept-product.md", "dept-market.md", "dept-hr.md", "dept-finance.md"]
                   if f not in files]
        print(f"漏掉的部门文档：{', '.join(missing) if missing else '（无）'}\n")

        context = "\n\n".join(f"[{i}] {e.snippet}" for i, e in enumerate(result.evidence, 1))
        client = OpenAI(base_url=cfg.agent_base_url, api_key=cfg.agent_api_key)
        response = client.chat.completions.create(
            model=cfg.agent_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是企业知识助手。只能根据给定的检索片段回答，不要使用任何外部知识。"},
                {"role": "user", "content": f"检索到的片段：\n\n{context}\n\n问题：{QUESTION}"},
            ],
        )
        answer = response.choices[0].message.content or ""

        print("─" * 68)
        print("模型基于这 5 条片段给出的回答：")
        print("─" * 68)
        print(answer)
        print("─" * 68)

        hedged = any(w in answer for w in
                     ["不确定", "可能不完整", "仅包含", "未提及", "无法确定", "不完整", "只涵盖", "所提供"])
        print(f"\n判定：")
        print(f"  漏召回了 {len(missing)} 个部门")
        print(f"  模型是否声明信息不完整：{'是' if hedged else '否 —— 静默给出了答案'}")
        if not hedged and missing:
            print(f"\n  ⚠ 这就是最危险的情况：答案带来源、有理有据，但少算了 {len(missing)} 个部门，")
            print(f"    用户无从察觉。检索的缺陷被生成层完美地掩盖了。")
        return 0

    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
