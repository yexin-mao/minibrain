"""手写 tool loop。

不上 LangChain/LangGraph：两个工具、单轮问答，一个 while 循环就够。
等真的需要持久化会话、长上下文摘要、并发写保护时再换框架也不迟 ——
那时候你会清楚自己为什么需要它，而不是因为教程里都这么写。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from .. import gateway
from ..config import get_config
from ..contracts import Evidence, ModuleError, UserContext
from .tools import TOOL_SCHEMAS, execute_tool

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        if not cfg.agent_configured:
            raise ModuleError(
                "AGENT_API_KEY 未配置，请在 .env 中填入", code="agent_not_configured", status=503
            )
        _client = OpenAI(base_url=cfg.agent_base_url, api_key=cfg.agent_api_key)
    return _client


@dataclass
class ToolCallTrace:
    name: str
    arguments: str
    result_preview: str


@dataclass
class AnswerResult:
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[ToolCallTrace] = field(default_factory=list)


def _system_prompt(user: UserContext) -> str:
    """两条链路的可用资源必须**对称**地注入。

    原来只注入了表结构（表名/列名/行数），文档侧只有一句泛泛描述。
    模型看得见表里有"组长"这一列，完全不知道文档里写了什么，于是两边都有的
    事实一律去查表——「文档·组织事实」类因此只有 1/6，三轮一致。

    消融实验（eval/PROMPT_ABLATION.md，43 题 × 4 版本 × 3 轮）证明：
      只加文件名        88.4% → 94.6%   目标类别 39% → 61%
      加文件名 + 标题    92.2%           目标类别 78%
      再加权威来源规则   97.7%           目标类别 100%（三轮 42/42/42）

    关键结论：**给资料 ≠ 给判断依据**。光列出文档修不好 doc-08——
    花名册里确实有"技术部"的 4 行记录，模型没理由怀疑它，
    除非明确告诉它那里面有记账用的辅助行。
    """
    table_schema = gateway.call("table-rag", "describe_schema", user)
    doc_catalog = gateway.call("vector-rag", "describe_corpus", user)
    modules = "\n".join(
        f"- {m.label}（{m.paradigm}）：{m.description}" for m in gateway.list_modules()
    )
    return f"""你是一个企业知识助手。你只能通过工具获取信息，不许凭记忆回答业务问题。

当前有两条相互独立的知识链路，检索范式不同，请按问题类型选择：
{modules}

选择原则：
- 问"怎么规定的""材料里怎么说"→ vector_search
- 问"多少""合计""平均""排名""按X分组"→ table_query。这类问题绝不能用向量检索猜，必须算。
- 一个问题同时涉及两类，就分别调用两个工具，再合并回答。

可检索的文档（只列出你有权访问的）：
{doc_catalog}

权威来源规则（重要）：同一个事实可能在文档和表格里都出现，此时以下面的规定为准：
- 组织架构、岗位职责、制度规定、项目信息 → **以文档为准**，用 vector_search
- 人数、金额、日期等需要统计计算的 → 以表格为准，用 table_query
- 表格里的行可能包含记账用的辅助行，不代表真实的组织单元。

可查询的数据表（只列出你有权访问的）：
{table_schema}

写 SQL 时：标识符一律用双引号，例如 SELECT "地区", sum("销售额") FROM t_xxxx GROUP BY "地区"。
只允许单条 SELECT。如果报错，读错误信息改写后重试，最多两次。

回答要求：
- 用中文，简洁。
- 每个事实性结论后面标注来源编号，例如 [1]，编号对应工具返回的片段序号。
- 工具没查到就直说"没有查到相关内容"，不要编。
- 当前用户：{user.username}（{"管理员" if user.is_admin else "普通用户"}）。
  你看到的内容已经按该用户权限过滤过，不要试图访问未列出的表。"""


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
            text, items = execute_tool(user, call.function.name, call.function.arguments)
            evidence.extend(items)
            trace.append(
                ToolCallTrace(
                    name=call.function.name,
                    arguments=call.function.arguments,
                    result_preview=text[:300] + ("…" if len(text) > 300 else ""),
                )
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

    return AnswerResult(
        answer="超过最大工具调用轮数仍未得到结论，请把问题问得更具体一些。",
        evidence=evidence,
        trace=trace,
    )
