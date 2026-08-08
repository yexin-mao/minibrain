"""LangGraph 版 agent。和手写的 `loop.py` **并存**，不是替换。

## 为什么要有这一版

`handwritten/agent_loop.py` 是 192 行手写 while 循环。它能跑，也能讲清 tool-calling 的机制，
但它有两个绕不过去的问题：

1. **没有会话。** 每次提问都是全新的，"上一轮说的那个项目"完全不支持。
   要自己做，就得写：历史存哪、并发怎么锁、超长了怎么裁。
2. **对标项目用的是框架。** `companybrain/apps/agent-gateway` 是
   LangChain 1.4 + LangGraph 1.3 + `langgraph-checkpoint-postgres`。

`loop.py` 开头那句注释说得没错——"等真的需要持久化会话、长上下文摘要、
并发写保护时再换框架"。**现在就是那个时候了。**

## 两版并存，不是骑墙

| | `loop.py`（手写） | `graph.py`（LangGraph） |
|---|---|---|
| 用途 | 评测基线、讲清机制 | **产品路径**：Web 问答走这条 |
| 会话 | 无 | Postgres checkpointer 持久化 |
| 停止判定 | 手写 for 循环 + 提示词 | 框架的 recursion_limit + 同一套提示词 |

保留手写版是有具体用处的：`scripts/eval_trajectory.py` 那套轨迹指标
（边际证据收益、冗余检索率）是针对 `ToolCallTrace` 写的。
**两版跑同一批用例，指标可以直接对比**——框架换来的是什么、代价是什么，
能拿数字说，而不是拿感觉说。

## 刻意复用的三样东西

**不能**因为换了框架就重写它们，否则历史结论全部作废：

1. `tools.py` 的工具描述 —— 描述文字直接影响工具选择准确率
2. `loop.py` 的 `_system_prompt` —— 消融实验（43 题 × 4 版本 × 3 轮）的产物，
   目标类别路由准确率 97.7% 就是它换来的
3. `AnswerResult` / `ToolCallTrace` 的形状 —— web 层和全部评测脚本都认它

换句话说：**换的是执行引擎，不是被验证过的那些决策。**
"""

from __future__ import annotations

import json
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent

from ..config import get_config
from ..contracts import Evidence, ModuleError, UserContext
from .prompt import system_prompt
from .types import AnswerResult, ToolCallTrace
from .tools import TOOL_SCHEMAS, execute_tool

_conn: Connection | None = None
_checkpointer: PostgresSaver | None = None


def _get_checkpointer() -> PostgresSaver:
    """会话状态存 PostgreSQL，和业务数据同库不同 schema。

    ★ 为什么不用内存 checkpointer：内存版进程一重启会话就没了，
      那等于没有会话。对标项目用的也是 postgres 版
      （`@langchain/langgraph-checkpoint-postgres`）。

    ★ `setup()` 会建它自己的表，是幂等的，所以每次拿都调一遍无所谓——
      和本项目 `schema.sql` 从空库幂等跑起是同一个约定。

    ★★ 这里踩过一个坑，值得记：
      第一版用的是 `PostgresSaver.from_conn_string(url)`，它返回的是一个
      **context manager**。我只保存了 `__enter__()` 的结果，没保存 CM 本身——
      于是 CM 被 GC 掉时顺手把连接关了，第二次调用报
      `OperationalError: the connection is closed`。

      **借来的连接，生命周期得自己盯着。** 所以改成显式建连接、显式持有。
      `autocommit=True` 是 checkpointer 的要求（它自己管事务边界）；
      `prepare_threshold=0` 关掉预备语句，避免和连接复用打架。
    """
    global _conn, _checkpointer
    if _checkpointer is None:
        _conn = Connection.connect(
            get_config().database_url,
            autocommit=True, prepare_threshold=0, row_factory=dict_row)
        _checkpointer = PostgresSaver(_conn)
        _checkpointer.setup()
    return _checkpointer


def _build_tools(user: UserContext, sink: list[Evidence]) -> list[StructuredTool]:
    """把 `tools.py` 里那两个工具包成 LangChain tool。

    ★★ 描述文字**逐字复用** `TOOL_SCHEMAS`，一个字都不改。

    工具描述直接决定工具选择准确率——`table_query` 那句
    "凡是需要合计、平均、排序、分组、筛选的问题都必须用这个工具，不要用向量检索猜"
    是路由消融实验调出来的。换框架时顺手"优化"一下措辞，
    等于把那次实验的结论悄悄作废，而且不会有任何报错提示你。

    ★ 每个工具闭包捕获 UserContext。**权限判断仍然发生在模块内部**，
      Agent 换成框架也绕不过去——这是本项目的硬约束，不因执行引擎而变。

    ★★ `sink` 是证据收集器，不是可选的装饰。

      LangChain 的 tool 只能返回一个字符串给模型，而本项目的
      `execute_tool` 同时返回 (给模型看的文本, 结构化证据)。
      证据要进 Web 的证据面板——"回答里每句话都要能指回一条证据"
      是这个项目的产品承诺，不能因为换了执行引擎就丢掉。

      框架不提供这条通路，所以用闭包捕获一个列表把证据接出来。
      **这类"框架表达不了的东西"正是换框架的真实代价**，
      写在这里而不是让它静默消失。
    """
    tools: list[StructuredTool] = []
    for schema in TOOL_SCHEMAS:
        fn_spec = schema["function"]
        name = fn_spec["name"]

        def _run(_name: str = name, **kwargs: Any) -> str:
            text, items = execute_tool(user, _name, json.dumps(kwargs, ensure_ascii=False))
            sink.extend(items)
            return text

        tools.append(StructuredTool(
            name=name,
            description=fn_spec["description"],
            args_schema=fn_spec["parameters"],       # 直接用原始 JSON Schema
            func=_run,
        ))
    return tools


def _to_trace(messages: list[Any]) -> list[ToolCallTrace]:
    """把 LangGraph 的 message 列表还原成本项目的 trace / evidence。

    ★ 这一层不是多余的适配代码，它是**让两版可比**的前提：
      `scripts/eval_trajectory.py` 的全部指标都建立在 ToolCallTrace 上
      （name / arguments / result_preview）。形状对齐了，
      同一套评测脚本才能同时量手写版和框架版。
    """
    trace: list[ToolCallTrace] = []
    pending: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                pending[call["id"]] = call
        elif isinstance(msg, ToolMessage):
            call = pending.get(msg.tool_call_id, {})
            content = str(msg.content)
            trace.append(ToolCallTrace(
                name=call.get("name", msg.name or "unknown"),
                arguments=json.dumps(call.get("args", {}), ensure_ascii=False),
                result_preview=content[:300] + ("…" if len(content) > 300 else ""),
            ))
    return trace


def answer(user: UserContext, question: str, *,
           session_id: str | None = None) -> AnswerResult:
    """回答一个问题。给了 session_id 就带上该会话的历史。

    ★ session_id 是**可选**的，缺省时行为和手写版完全一致（单轮）。
      这样评测脚本不用改就能跑框架版，两版数字直接可比。
    """
    cfg = get_config()
    if not cfg.agent_configured:
        raise ModuleError("AGENT_API_KEY 未配置，请在 .env 中填入",
                          code="agent_not_configured", status=503)

    model = ChatOpenAI(
        base_url=cfg.agent_base_url, api_key=cfg.agent_api_key,
        model=cfg.agent_model, temperature=cfg.agent_temperature,
        # 超时和重试沿用同一套配置。踩过 13 小时挂死那次之后，
        # 任何出站调用都必须有上界，换框架不改变这条。
        timeout=cfg.llm_timeout_seconds, max_retries=cfg.llm_max_retries,
    )

    evidence: list[Evidence] = []
    agent = create_react_agent(
        model,
        _build_tools(user, evidence),
        # 系统提示词逐字复用手写版。它是消融实验的产物，不因换框架而重写。
        prompt=SystemMessage(content=system_prompt(user)),
        checkpointer=_get_checkpointer() if session_id else None,
    )

    config: dict[str, Any] = {
        # recursion_limit 对应手写版的 agent_max_steps。
        # ★ 语义不完全一样：LangGraph 数的是**图的步数**（模型一步 + 工具一步），
        #   手写版数的是**工具调用轮数**。所以这里乘 2 再加一点余量，
        #   否则框架版会比手写版更早被掐断，两版就没法比了。
        "recursion_limit": cfg.agent_max_steps * 2 + 2,
    }
    if session_id:
        config["configurable"] = {"thread_id": session_id}

    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config)
    messages = result["messages"]
    trace = _to_trace(messages)

    final = next((m for m in reversed(messages)
                  if isinstance(m, AIMessage) and not m.tool_calls), None)
    return AnswerResult(
        answer=(final.content if final else "（模型没有返回内容）"),
        evidence=evidence,
        trace=trace,
    )
