"""Run Trace 的持久化、统计和用户隔离。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from minibrain.agent import graph
from minibrain.agent.citations import ClaimOutput, StructuredAnswer
from minibrain.agent.context import ContextAssembler
from minibrain.agent.confidence import SAFE_REFUSAL
from minibrain.agent.graph import HistoryWindowMiddleware, _token_usage, _trim_history
from minibrain.contracts import Evidence, ModuleError, NotFound, UserContext
from minibrain.db import observability_db
from minibrain.observability import core


def test_query_key_only_folds_cosmetic_differences():
    assert graph._query_key("算法组 属于哪个部门？") == graph._query_key(
        "算法组属于哪个部门?"
    )
    assert graph._query_key("算法组负责人") != graph._query_key("算法组所属部门")


def test_stale_running_run_is_archived_but_fresh_run_is_kept(alice):
    # 开发库可能保留上一次被中断测试的 stale run；先执行一次产品恢复动作，
    # 后面的计数才只属于本测试创建的记录。
    core.archive_stale_runs(60)
    stale = core.start_run(alice, "旧问题", session_id=None, model="test")
    fresh = core.start_run(alice, "新问题", session_id=None, model="test")
    with observability_db() as cur:
        cur.execute(
            "UPDATE runs SET started_at = now() - interval '1 hour' WHERE id = %s",
            (stale,),
        )

    assert core.archive_stale_runs(60) == 1
    with observability_db() as cur:
        cur.execute("SELECT id, status, error FROM runs WHERE id IN (%s, %s)",
                    (stale, fresh))
        rows = {str(row["id"]): row for row in cur.fetchall()}
    assert rows[stale]["status"] == "failed"
    assert rows[stale]["error"]["code"] == "stale_run_archived"
    assert rows[fresh]["status"] == "running"


def test_repeated_vector_query_is_blocked_before_retrieval(monkeypatch):
    calls = []

    def fake_execute(user, name, arguments, *, evidence_prefix=None):
        calls.append(arguments)
        item = Evidence(
            module="vector-rag", source_name="知识库", location="team.md #1",
            snippet="算法组属于技术部", evidence_id=f"{evidence_prefix}.1",
        )
        return "unused", [item]

    monkeypatch.setattr(graph, "execute_tool", fake_execute)
    assembler = ContextAssembler(
        token_budget=1000, evidence_limit=12, candidate_pool=12)
    tools = graph._build_tools(
        UserContext("u1", "alice", False), [], assembler)
    vector = next(tool for tool in tools if tool.name == "vector_search")

    first = vector.invoke({"query": "算法组 属于哪个部门？"})
    repeated = vector.invoke({"query": "算法组属于哪个部门?"})

    assert "算法组属于技术部" in first
    assert "已经执行过的查询重复" in repeated
    assert len(calls) == 1


def test_successful_run_is_visible_to_owner_and_admin(alice, bob, admin):
    run_id = core.start_run(
        alice, "报销标准是什么？", session_id="session-1", model="test-model")
    core.finish_success(
        run_id,
        answer="每天 500 元。",
        latency_ms=123,
        usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        tool_calls=[{
            "name": "vector_search", "arguments": "{}",
            "result_preview": "制度片段", "step": 1,
        }],
        evidence=[{
            "module": "vector-rag", "source_name": "制度",
            "location": "policy.md#1", "snippet": "每天 500 元", "score": 0.9,
        }],
        claims=[{
            "text": "每天 500 元", "evidence_ids": ["E1.1"],
            "valid_evidence_ids": ["E1.1"], "invalid_evidence_ids": [],
        }],
        citation_metrics={"citation_coverage": 1.0, "citation_validity": 1.0},
        context_decisions=[{
            "evidence_id": "E1.1", "selected": True, "reason": "selected",
        }],
        context_metrics={
            "tokenizer": "cl100k_base", "token_budget": 4000,
            "context_tokens": 100, "selected_count": 1,
        },
        confidence_report={"status": "deterministic_pass", "blocked": False},
    )

    owner_run = core.get_run(alice, run_id)
    assert owner_run["status"] == "succeeded"
    assert owner_run["total_tokens"] == 18
    assert owner_run["tool_calls"][0]["name"] == "vector_search"
    assert owner_run["citation_metrics"]["citation_coverage"] == 1.0
    assert owner_run["context_metrics"]["context_tokens"] == 100
    assert owner_run["confidence_report"]["status"] == "deterministic_pass"
    assert any(str(item["id"]) == run_id for item in core.list_runs(alice))
    assert core.get_run(admin, run_id)["question"] == "报销标准是什么？"

    with pytest.raises(NotFound):
        core.get_run(bob, run_id)


def test_failed_run_records_stable_error(alice):
    run_id = core.start_run(
        alice, "失败问题", session_id=None, model="test-model")
    core.finish_failure(
        run_id,
        error=ModuleError("上游超时", code="upstream_timeout", status=503),
        latency_ms=456,
    )

    row = core.get_run(alice, run_id)
    assert row["status"] == "failed"
    assert row["latency_ms"] == 456
    assert row["error"] == {"code": "upstream_timeout", "message": "上游超时"}


def test_run_can_be_found_by_session_scoped_request_id(alice):
    request_id = "11111111-1111-4111-8111-111111111111"
    run_id = core.start_run(
        alice, "幂等问题", session_id="thread-1", request_id=request_id,
        model="test-model",
    )

    row = core.find_run_by_request(
        alice, session_id="thread-1", request_id=request_id)

    assert str(row["id"]) == run_id
    assert row["question"] == "幂等问题"


def test_feedback_upserts_and_exports_negative_run(alice):
    run_id = core.start_run(
        alice, "反馈问题", session_id="feedback-session", model="test-model")
    core.finish_success(
        run_id, answer="旧答案", latency_ms=10,
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        tool_calls=[], evidence=[{
            "evidence_id": "E1.1", "module": "vector-rag",
            "source_name": "制度", "location": "a.md#1", "snippet": "证据",
        }], claims=[], citation_metrics={}, context_decisions=[], context_metrics={},
    )

    first = core.save_feedback(
        alice, run_id, rating=-1, reason="incorrect", note="金额错了")
    assert first["rating"] == -1
    assert core.get_feedback(alice, run_id)["note"] == "金额错了"

    samples = core.export_regression_samples(alice)
    sample = next(item for item in samples if item["run"]["id"] == run_id)
    assert sample["case_id"] == f"feedback-{run_id}"
    assert sample["question"] == "反馈问题"
    assert sample["expected_answer"] is None
    assert sample["review"]["reason"] == "incorrect"

    # 同一用户/运行再次点击会修正旧反馈，不会重复造样本。
    second = core.save_feedback(alice, run_id, rating=1, note="重新确认后正确")
    assert second["id"] == first["id"]
    assert core.get_feedback(alice, run_id)["reason"] is None
    assert not any(
        item["run"]["id"] == run_id
        for item in core.export_regression_samples(alice)
    )


def test_feedback_cannot_be_submitted_for_another_users_run(alice, bob):
    run_id = core.start_run(
        alice, "私有问题", session_id=None, model="test-model")
    core.finish_success(
        run_id, answer="答案", latency_ms=1,
        usage={}, tool_calls=[], evidence=[], claims=[], citation_metrics={},
        context_decisions=[], context_metrics={},
    )
    with pytest.raises(ModuleError) as caught:
        core.save_feedback(bob, run_id, rating=-1, reason="incorrect")
    assert caught.value.code == "not_found"


def test_metrics_aggregate_real_run_outcomes(alice):
    before = core.metrics_overview(alice, hours=24)
    success_id = core.start_run(
        alice, "监控成功", session_id=None, model="test-model")
    core.finish_success(
        success_id, answer="答案", latency_ms=120,
        usage={"total_tokens": 9}, tool_calls=[], evidence=[], claims=[],
        citation_metrics={"citation_coverage": 0.5},
        context_decisions=[], context_metrics={},
        raw_answer="未支持答案",
        confidence_report={"status": "blocked", "blocked": True},
    )
    failure_id = core.start_run(
        alice, "监控失败", session_id=None, model="test-model")
    core.finish_failure(
        failure_id, error=ModuleError("超时", code="test_timeout"), latency_ms=300)

    after = core.metrics_overview(alice, hours=24)

    assert after["total_runs"] == before["total_runs"] + 2
    assert after["succeeded_runs"] == before["succeeded_runs"] + 1
    assert after["failed_runs"] == before["failed_runs"] + 1
    assert after["total_tokens"] == before["total_tokens"] + 9
    assert after["confidence_blocked_runs"] == before["confidence_blocked_runs"] + 1
    assert after["latency_p50_ms"] is not None
    assert any(item["code"] == "test_timeout" for item in after["failure_codes"])


def test_token_usage_only_counts_latest_turn():
    messages = [
        HumanMessage(content="上一问"),
        AIMessage(content="上一答", usage_metadata={
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        }),
        HumanMessage(content="这一问"),
        AIMessage(content="调用工具", usage_metadata={
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
        }),
        AIMessage(content="这一答", response_metadata={"token_usage": {
            "prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20,
        }}),
    ]

    assert _token_usage(messages) == {
        "input_tokens": 25, "output_tokens": 7, "total_tokens": 32,
    }


def test_history_window_keeps_complete_recent_turns():
    messages = []
    for index in range(4):
        messages.extend([
            HumanMessage(content=f"问题 {index}"),
            AIMessage(content=f"回答 {index}"),
        ])

    trimmed = _trim_history(messages, max_turns=2, token_budget=10_000)

    assert [message.content for message in trimmed] == [
        "问题 2", "回答 2", "问题 3", "回答 3",
    ]


def test_history_window_always_keeps_current_tool_protocol():
    messages = [
        HumanMessage(content="很长的旧问题 " * 100),
        AIMessage(content="很长的旧回答 " * 100),
        HumanMessage(content="当前问题"),
        AIMessage(content="", tool_calls=[{
            "id": "call-1", "name": "vector_search", "args": {"query": "当前问题"},
        }]),
        ToolMessage(content="当前证据", tool_call_id="call-1", name="vector_search"),
    ]

    trimmed = _trim_history(messages, max_turns=6, token_budget=20)

    assert trimmed == messages[2:]
    assert isinstance(trimmed[-1], ToolMessage)


def test_agent_persists_success_and_returns_run_id(alice, monkeypatch):
    class Config:
        agent_model = "test-model"
        agent_configured = True
        agent_base_url = "https://example.invalid"
        agent_api_key = "test-key"
        agent_temperature = 0
        llm_timeout_seconds = 1
        llm_max_retries = 0
        agent_max_steps = 3
        agent_evidence_token_budget = 4000
        agent_evidence_limit = 12
        agent_context_candidate_pool = 12
        agent_history_max_turns = 6
        agent_history_token_budget = 8000

    class FakeAgent:
        def stream(self, payload, config, stream_mode):
            yield {
                "messages": [
                    payload["messages"][0],
                    AIMessage(content="", usage_metadata={
                        "input_tokens": 9, "output_tokens": 3, "total_tokens": 12,
                    }),
                ],
                "structured_response": StructuredAnswer(
                    answer="测试答案",
                    claims=[ClaimOutput(text="测试结论", evidence_ids=[])],
                ),
            }

    saved = {}
    created = {}
    monkeypatch.setattr(graph, "get_config", Config)
    monkeypatch.setattr(graph, "ChatOpenAI", lambda **kwargs: object())
    def fake_create_agent(**kwargs):
        created.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph, "system_prompt", lambda user: "prompt")
    monkeypatch.setattr(graph, "_start_trace", lambda *args, **kwargs: "run-123")
    monkeypatch.setattr(
        graph, "_finish_trace_success",
        lambda run_id, **kwargs: saved.update(run_id=run_id, **kwargs),
    )

    result = graph.answer(alice, "测试问题")

    assert result.answer == SAFE_REFUSAL
    assert result.run_id == "run-123"
    assert result.claims[0].text == "测试结论"
    assert isinstance(created["response_format"], graph.ToolStrategy)
    assert isinstance(created["middleware"][0], HistoryWindowMiddleware)
    tool_limits = created["middleware"][1:]
    assert [item.tool_name for item in tool_limits] == [
        "vector_search", "table_query",
    ]
    assert all(item.run_limit == 3 for item in tool_limits)
    assert saved["usage"]["total_tokens"] == 12
    assert result.confidence_report is not None
    assert result.confidence_report.blocked is True
    assert saved["answer"] == SAFE_REFUSAL
    assert saved["raw_answer"] == "测试答案"
    assert saved["confidence_report"]["status"] == "blocked"
    assert saved["citation_metrics"]["citation_coverage"] == 0.0
    assert saved["context_metrics"]["token_budget"] == 4000


def test_agent_persists_configuration_failure(alice, monkeypatch):
    class Config:
        agent_model = "test-model"
        agent_configured = False

    saved = {}
    monkeypatch.setattr(graph, "get_config", Config)
    monkeypatch.setattr(graph, "_start_trace", lambda *args, **kwargs: "run-456")
    monkeypatch.setattr(
        graph, "_finish_trace_failure",
        lambda run_id, **kwargs: saved.update(run_id=run_id, **kwargs),
    )

    with pytest.raises(ModuleError, match="AGENT_API_KEY"):
        graph.answer(alice, "测试问题")

    assert saved["run_id"] == "run-456"
    assert saved["error"].code == "agent_not_configured"


def test_recursion_limit_forces_structured_final_from_last_state():
    tool_call = {
        "name": "vector_search", "args": {"query": "Beta"}, "id": "call-1",
        "type": "tool_call",
    }
    raw = AIMessage(content="", usage_metadata={
        "input_tokens": 8, "output_tokens": 2, "total_tokens": 10,
    })

    class FakeAgent:
        def stream(self, payload, config, stream_mode):
            yield {"messages": [
                payload["messages"][0],
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content="没有查到 Beta 预算", tool_call_id="call-1",
                            name="vector_search"),
            ]}
            raise graph.GraphRecursionError("limit")

    class StructuredModel:
        def invoke(self, messages):
            assert isinstance(messages[-1], HumanMessage)
            return {
                "parsed": StructuredAnswer(answer="没有查到预算。", claims=[]),
                "raw": raw, "parsing_error": None,
            }

    class FakeModel:
        def with_structured_output(self, schema, **kwargs):
            assert schema is StructuredAnswer
            assert kwargs == {"method": "function_calling", "include_raw": True}
            return StructuredModel()

    result, forced_raw, pending = graph._invoke_with_forced_final(
        FakeAgent(), FakeModel(), {"messages": [HumanMessage("预算？")]},
        {"recursion_limit": 2},
    )

    assert result["structured_response"].answer == "没有查到预算。"
    assert forced_raw is raw
    assert pending == []
