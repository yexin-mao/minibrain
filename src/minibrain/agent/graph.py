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
| 停止判定 | 手写 for 循环 + 强制收口 | 提示词 + 单工具调用上限 + recursion_limit |

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
import time
import unicodedata
from dataclasses import asdict
from itertools import count
from threading import Lock
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy

from ..config import get_config
from ..contracts import Evidence, ModuleError, UserContext
from ..observability import core as observability
from .citations import StructuredAnswer, complete_truncated_answer, validate_claims
from .confidence import apply_confidence_gate, assess_answer_support
from .context import ContextAssembler
from .session_control import normalize_request_id, previous_result, session_lock
from .prompt import system_prompt
from .types import AnswerResult, ToolCallTrace
from .tools import TOOL_SCHEMAS, execute_tool, format_evidence

_conn: Connection | None = None
_checkpointer: PostgresSaver | None = None


def _trim_history(messages: list[Any], *, max_turns: int,
                  token_budget: int) -> list[Any]:
    """按完整用户 turn 保留最近历史，永不拆断 tool call 协议。

    当前 turn 无条件保留；预算只决定还能带多少旧 turn。完整状态仍由
    checkpointer 保存，本函数只裁剪一次模型请求的临时视图。
    """
    starts = [i for i, message in enumerate(messages)
              if isinstance(message, HumanMessage)]
    if len(starts) <= 1:
        return list(messages)
    turns = [
        messages[start:(starts[index + 1] if index + 1 < len(starts) else len(messages))]
        for index, start in enumerate(starts)
    ]
    selected = [turns[-1]]
    for turn in reversed(turns[:-1]):
        if len(selected) >= max_turns:
            break
        candidate = [*turn, *(message for kept in selected for message in kept)]
        if count_tokens_approximately(candidate) > token_budget:
            break
        selected.insert(0, turn)
    return [message for turn in selected for message in turn]


class HistoryWindowMiddleware(AgentMiddleware):
    """保留持久化全历史，只限制送入单次模型调用的上下文窗口。"""

    def __init__(self, *, max_turns: int, token_budget: int) -> None:
        self.max_turns = max_turns
        self.token_budget = token_budget

    def wrap_model_call(self, request: ModelRequest,
                        handler: Any) -> ModelResponse | AIMessage:
        trimmed = _trim_history(
            list(request.messages), max_turns=self.max_turns,
            token_budget=self.token_budget)
        return handler(request.override(messages=trimmed))


def _query_key(query: str) -> str:
    """只折叠无语义差异，避免把不同关系的多跳查询误判成重复。"""
    normalised = unicodedata.normalize("NFKC", query).casefold()
    return "".join(
        char for char in normalised
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "S"))
    )


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


def _build_tools(user: UserContext, sink: list[Evidence],
                 assembler: ContextAssembler) -> list[StructuredTool]:
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
    # itertools.count 的 next 在 CPython 中是原子的。并行 tool call 会先各自领取
    # 一个调用号，因此即使完成顺序相反，也不会生成重复 evidence_id。
    call_numbers = count(1)
    seen_vector_queries: set[str] = set()
    query_lock = Lock()
    for schema in TOOL_SCHEMAS:
        fn_spec = schema["function"]
        name = fn_spec["name"]

        def _run(_name: str = name, **kwargs: Any) -> str:
            call_number = next(call_numbers)
            if _name == "vector_search":
                query = str(kwargs.get("query", ""))
                filename = str(kwargs.get("filename", ""))
                source = str(kwargs.get("source", ""))
                key = f"{_query_key(query)}\0{filename.casefold()}\0{source.casefold()}"
                with query_lock:
                    if key and key in seen_vector_queries:
                        return (
                            f"检索 query={query!r} 与本轮已经执行过的查询重复，"
                            "没有产生新证据。请改查由现有证据得到的中间实体及缺失关系；"
                            "若没有新的中间实体，请停止检索并说明缺失信息。"
                        )
                    if key:
                        seen_vector_queries.add(key)
                # top_k 是候选召回量，不再等同于最终上下文条数。
                kwargs["top_k"] = assembler.candidate_pool
            text, items = execute_tool(
                user, _name, json.dumps(kwargs, ensure_ascii=False),
                evidence_prefix=f"E{call_number}",
            )
            if not items:
                return text
            # 第一跳不能独占 12 条/4000 token 的全局预算。每次最多 3 条，
            # 为“实体发现 → 长文档命中 → 文档内章节检索”三阶段保留空间。
            selected = assembler.add(items, max_items=3)
            sink.extend(selected)
            if not selected:
                return "候选内容均因重复或上下文预算限制未进入本轮证据。"
            return format_evidence(selected)

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
    # id → (调用信息, 它属于第几轮)
    pending: dict[str, tuple[dict, int]] = {}
    step = 0

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # ★ 一条 AIMessage = 一轮。里面可能有多个 tool_calls（并行调用），
            #   它们共享同一个 step —— 这样 max(step) 才是真实的轮数。
            step += 1
            for call in msg.tool_calls:
                if call.get("name") == StructuredAnswer.__name__:
                    continue
                pending[call["id"]] = (call, step)
        elif isinstance(msg, ToolMessage):
            if msg.name == StructuredAnswer.__name__:
                continue
            call, call_step = pending.get(msg.tool_call_id, ({}, step))
            content = str(msg.content)
            trace.append(ToolCallTrace(
                name=call.get("name", msg.name or "unknown"),
                arguments=json.dumps(call.get("args", {}), ensure_ascii=False),
                result_preview=content[:300] + ("…" if len(content) > 300 else ""),
                step=call_step,
            ))
    return trace


def _current_turn(messages: list[Any]) -> list[Any]:
    """只取最后一条用户消息之后的消息，避免多轮会话重复统计历史 token。"""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index + 1:]
    return messages


def _token_usage(messages: list[Any]) -> dict[str, int]:
    """兼容 LangChain 标准 usage_metadata 和供应商原始 token_usage。"""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for msg in _current_turn(messages):
        if not isinstance(msg, AIMessage):
            continue
        metadata = getattr(msg, "usage_metadata", None)
        if metadata:
            usage["input_tokens"] += int(metadata.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(metadata.get("output_tokens", 0) or 0)
            usage["total_tokens"] += int(metadata.get("total_tokens", 0) or 0)
            continue
        raw = (getattr(msg, "response_metadata", {}) or {}).get("token_usage", {})
        usage["input_tokens"] += int(raw.get("prompt_tokens", 0) or 0)
        usage["output_tokens"] += int(raw.get("completion_tokens", 0) or 0)
        usage["total_tokens"] += int(raw.get("total_tokens", 0) or 0)
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left.get(key, 0) + right.get(key, 0)
            for key in ("input_tokens", "output_tokens", "total_tokens")}


def _pending_tool_messages(messages: list[Any]) -> list[ToolMessage]:
    """补齐递归上限恰好截在 tool call 后时尚未返回的协议消息。"""
    answered = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    pending = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call["id"] not in answered:
                pending.append(ToolMessage(
                    content="工具调用上限已到，本次调用未执行。请基于已有证据作答。",
                    tool_call_id=call["id"], name=call.get("name"), status="error",
                ))
    return pending


def _invoke_with_forced_final(agent: Any, model: Any, payload: dict[str, Any],
                              config: dict[str, Any]) -> tuple[dict[str, Any],
                                                               AIMessage | None,
                                                               list[ToolMessage]]:
    """保留最后图状态；到递归上限后移除业务工具并强制结构化收口。

    `invoke()` 抛 GraphRecursionError 时不会返回已积累的 messages。使用 values
    stream 才能保住它们和工具证据，这正是旧手写版强制收口所依赖的信息。
    """
    state: dict[str, Any] | None = None
    try:
        for value in agent.stream(payload, config, stream_mode="values"):
            state = value
    except GraphRecursionError:
        if state is None or not state.get("messages"):
            raise
        messages = list(state["messages"])
        pending = _pending_tool_messages(messages)
        forced = model.with_structured_output(
            StructuredAnswer, method="function_calling", include_raw=True,
        ).invoke([
            *messages,
            *pending,
            HumanMessage(content=(
                "检索调用已达上限。不要再调用任何检索工具；请只基于上面已有证据"
                "提交最终结构化回答。能确定的直接回答，不能确定的明确说明缺什么。"
            )),
        ])
        parsed = forced.get("parsed")
        parsing_error = forced.get("parsing_error")
        if parsing_error is not None or parsed is None:
            raise RuntimeError(f"强制结构化收口失败：{parsing_error}")
        if isinstance(parsed, dict):
            parsed = StructuredAnswer.model_validate(parsed)
        if not isinstance(parsed, StructuredAnswer):
            raise RuntimeError("强制结构化收口没有返回 StructuredAnswer")
        raw = forced.get("raw")
        state = dict(state)
        state["structured_response"] = parsed
        return state, raw if isinstance(raw, AIMessage) else None, pending
    if state is None:
        raise RuntimeError("Agent 没有返回任何状态")
    return state, None, []


def _start_trace(user: UserContext, question: str, *, session_id: str | None,
                 model: str, request_id: str | None = None) -> str | None:
    """可观测性是旁路能力：记录失败不能让主问答失败。"""
    try:
        return observability.start_run(
            user, question, session_id=session_id, model=model,
            request_id=request_id)
    except Exception:
        return None


def _finish_trace_success(run_id: str | None, **kwargs: Any) -> None:
    if run_id is None:
        return
    try:
        observability.finish_success(run_id, **kwargs)
    except Exception:
        pass


def _finish_trace_failure(run_id: str | None, **kwargs: Any) -> None:
    if run_id is None:
        return
    try:
        observability.finish_failure(run_id, **kwargs)
    except Exception:
        pass


def _answer_once(user: UserContext, question: str, *,
                 session_id: str | None = None,
                 request_id: str | None = None) -> AnswerResult:
    """回答一个问题。给了 session_id 就带上该会话的历史。

    ★ session_id 是**可选**的，缺省时行为和手写版完全一致（单轮）。
      这样评测脚本不用改就能跑框架版，两版数字直接可比。
    """
    cfg = get_config()
    started = time.perf_counter()
    run_id = _start_trace(
        user, question, session_id=session_id, model=cfg.agent_model,
        request_id=request_id)
    try:
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
        assembler = ContextAssembler(
            token_budget=cfg.agent_evidence_token_budget,
            evidence_limit=cfg.agent_evidence_limit,
            candidate_pool=cfg.agent_context_candidate_pool,
        )
        # ★★ 用 langchain.agents.create_agent，不是已弃用的 prebuilt 工厂。
        # 这里保留 middleware 扩展点，同时逐字复用已经评测过的系统提示词。
        agent = create_agent(
            model=model,
            tools=_build_tools(user, evidence, assembler),
            system_prompt=system_prompt(user),
            # 旧手写 Agent 到 max_steps 后会移除工具、强制基于已有证据收口；
            # LangGraph 迁移时只留下 recursion_limit，实测会在 7/8 个 smoke
            # 用例上直接抛 GraphRecursionError。这里分别限制两个业务工具，
            # 超限调用会变成 ToolMessage，模型仍可提交 StructuredAnswer。
            # 不能用 tool_name=None：那会把结构化输出自身也算成业务工具调用。
            middleware=[
                HistoryWindowMiddleware(
                    max_turns=cfg.agent_history_max_turns,
                    token_budget=cfg.agent_history_token_budget,
                ),
                ToolCallLimitMiddleware(
                    tool_name="vector_search", run_limit=cfg.agent_max_steps,
                    exit_behavior="continue"),
                ToolCallLimitMiddleware(
                    tool_name="table_query", run_limit=cfg.agent_max_steps,
                    exit_behavior="continue"),
            ],
            checkpointer=_get_checkpointer() if session_id else None,
            # ToolStrategy 对 OpenAI-compatible 供应商也可用，不依赖厂商原生
            # json_schema。它取代原本的自由文本最终回答，不增加 evidence selector。
            response_format=ToolStrategy(
                StructuredAnswer,
                tool_message_content="结构化回答已提交。",
            ),
        )

        config: dict[str, Any] = {
            # LangGraph 数图步数（模型一步 + 工具一步），手写版数工具轮数。
            # +4 给「超限工具被阻止 → 模型提交结构化最终答案」预留空间；
            # 真正的业务调用数由上面的 middleware 限制，不因这里变大而放宽。
            "recursion_limit": cfg.agent_max_steps * 2 + 4,
        }
        if session_id:
            config["configurable"] = {"thread_id": session_id}

        result, forced_raw, pending_messages = _invoke_with_forced_final(
            agent, model, {"messages": [HumanMessage(content=question)]}, config)
        messages = result["messages"]
        trace = _to_trace(_current_turn(messages))

        structured = result.get("structured_response")
        if isinstance(structured, dict):
            structured = StructuredAnswer.model_validate(structured)
        if isinstance(structured, StructuredAnswer):
            answer_text = complete_truncated_answer(
                structured.answer, structured.claims)
            raw_claims = structured.claims
        else:
            # 兼容旧 checkpoint、测试替身和不支持 structured tool 的供应商。
            # 降级只影响 claim 指标，不让用户丢掉已经生成的答案。
            final = next((m for m in reversed(messages)
                          if isinstance(m, AIMessage) and not m.tool_calls), None)
            answer_text = str(final.content) if final else "（模型没有返回内容）"
            raw_claims = []
        raw_answer_text = answer_text
        claims, citation_metrics = validate_claims(raw_claims, evidence)
        confidence_report = assess_answer_support(
            claims, citation_metrics, evidence)
        answer_text = apply_confidence_gate(answer_text, confidence_report)
        if forced_raw is not None and session_id:
            # 强制收口发生在 agent 图之外；把协议补全消息和最终文本写回 checkpoint，
            # 否则下一轮会话会看到一个没有结尾的 tool call。
            agent.update_state(config, {
                "messages": [
                    *pending_messages,
                    AIMessage(content=answer_text),
                ],
            })
        context_metrics = assembler.metrics()
        latency_ms = round((time.perf_counter() - started) * 1000)
        usage = _token_usage(messages)
        if forced_raw is not None:
            usage = _add_usage(usage, _token_usage([forced_raw]))
        _finish_trace_success(
            run_id,
            answer=answer_text,
            latency_ms=latency_ms,
            usage=usage,
            tool_calls=[asdict(item) for item in trace],
            evidence=[asdict(item) for item in evidence],
            claims=[asdict(item) for item in claims],
            citation_metrics=asdict(citation_metrics),
            context_decisions=[asdict(item) for item in assembler.decisions],
            context_metrics=asdict(context_metrics),
            # 正常回答不重复存两份；只有被拦截时保留原文供 run detail 审计。
            raw_answer=raw_answer_text if confidence_report.blocked else None,
            confidence_report=asdict(confidence_report),
        )
        return AnswerResult(
            answer=answer_text,
            evidence=evidence,
            trace=trace,
            run_id=run_id,
            claims=claims,
            citation_metrics=citation_metrics,
            context_decisions=list(assembler.decisions),
            context_metrics=context_metrics,
            forced_final=forced_raw is not None,
            confidence_report=confidence_report,
        )
    except Exception as exc:
        _finish_trace_failure(
            run_id, error=exc,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        raise


def answer(user: UserContext, question: str, *, session_id: str | None = None,
           request_id: str | None = None) -> AnswerResult:
    """同一 thread 跨进程串行；相同 request_id 成功重试不重复调用模型。"""
    normalized_request = normalize_request_id(request_id)
    if session_id is None:
        return _answer_once(user, question, request_id=normalized_request)

    with session_lock(session_id):
        if normalized_request is not None:
            existing = previous_result(
                user, session_id=session_id, request_id=normalized_request,
                question=question)
            if existing is not None:
                return existing
        return _answer_once(
            user, question, session_id=session_id,
            request_id=normalized_request)
