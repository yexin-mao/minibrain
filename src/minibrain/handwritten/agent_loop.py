"""手写 tool loop。

不上 LangChain/LangGraph：两个工具、单轮问答，一个 while 循环就够。
等真的需要持久化会话、长上下文摘要、并发写保护时再换框架也不迟 ——
那时候你会清楚自己为什么需要它，而不是因为教程里都这么写。
"""

from __future__ import annotations


from openai import OpenAI

from ..config import get_config
from ..agent.prompt import system_prompt as _system_prompt
from ..agent.types import AnswerResult, ToolCallTrace
from ..contracts import Evidence, ModuleError, UserContext
from ..agent.tools import TOOL_SCHEMAS, execute_tool

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        if not cfg.agent_configured:
            raise ModuleError(
                "AGENT_API_KEY 未配置，请在 .env 中填入", code="agent_not_configured", status=503
            )
        # timeout / max_retries 见 config.py 里那段注释：不设的话 SDK 默认
        # 600 秒 + 2 次重试，一次挂死能拖十几个小时且看不出在挂。
        _client = OpenAI(base_url=cfg.agent_base_url, api_key=cfg.agent_api_key,
                         timeout=cfg.llm_timeout_seconds,
                         max_retries=cfg.llm_max_retries)
    return _client


def answer(user: UserContext, question: str) -> AnswerResult:
    cfg = get_config()
    client = _get_client()

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(user)},
        {"role": "user", "content": question},
    ]
    evidence: list[Evidence] = []
    trace: list[ToolCallTrace] = []

    for _ in range(cfg.agent_max_steps):
        try:
            response = client.chat.completions.create(
                model=cfg.agent_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=cfg.agent_temperature,
            )
        except Exception as exc:
            raise ModuleError(
                f"Agent 模型调用失败：{type(exc).__name__}", code="agent_failed", status=502
            ) from exc

        message = response.choices[0].message

        if not message.tool_calls:
            return AnswerResult(answer=message.content or "（模型没有返回内容）", evidence=evidence, trace=trace)

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for call in message.tool_calls:
            text, items = execute_tool(
                user, call.function.name, call.function.arguments,
                evidence_prefix=f"E{len(trace) + 1}",
            )
            evidence.extend(items)
            trace.append(
                ToolCallTrace(
                    name=call.function.name,
                    arguments=call.function.arguments,
                    result_preview=text[:300] + ("…" if len(text) > 300 else ""),
                )
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

    # 轮数用尽。原来直接返回一句"超过最大工具调用轮数"——**那是最糟的输出**：
    # 它把已经检索到的全部内容扔掉了，用户什么也没得到。
    #
    # 实测（eval/RESULTS.md 探针五）：跑飞时它已经查了 6~8 次，手里有大量证据，
    # 只是没能自己收口。所以再问模型一次，**但不给工具**——强制它基于已有信息作答。
    messages.append({
        "role": "user",
        "content": "检索轮数已达上限，不要再调用工具。请基于上面已经检索到的内容作答："
                   "能确定的部分直接给出，不能确定的明确说明缺什么。",
    })
    try:
        final = client.chat.completions.create(
            model=cfg.agent_model, messages=messages, temperature=cfg.agent_temperature,
        )
        text = final.choices[0].message.content
    except Exception:                                   # noqa: BLE001
        text = None

    return AnswerResult(
        answer=text or "检索轮数已达上限，且未能基于已有内容归纳出结论。",
        evidence=evidence,
        trace=trace,
    )
