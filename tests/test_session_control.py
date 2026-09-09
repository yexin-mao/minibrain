from contextlib import contextmanager
from uuid import uuid4

import pytest

from minibrain.agent import graph
from minibrain.agent.session_control import (
    normalize_request_id,
    previous_result,
    session_lock,
    valid_chat_id,
)
from minibrain.agent.types import AnswerResult
from minibrain.contracts import ModuleError, UserContext


def _user():
    return UserContext(str(uuid4()), "alice", False)


def test_chat_id_must_belong_to_current_user_and_contain_uuid4():
    user = _user()
    valid = f"{user.user_id}:{uuid4().hex}"

    assert valid_chat_id(user, valid) is True
    assert valid_chat_id(user, f"{uuid4()}:{uuid4().hex}") is False
    assert valid_chat_id(user, f"{user.user_id}:not-a-uuid") is False


def test_request_id_is_canonicalized_and_invalid_value_rejected():
    request_id = uuid4()
    assert normalize_request_id(request_id.hex) == str(request_id)
    with pytest.raises(ModuleError) as caught:
        normalize_request_id("retry-1")
    assert caught.value.code == "invalid_request_id"


def test_previous_successful_request_is_reconstructed(monkeypatch):
    user = _user()
    run_id = uuid4()
    monkeypatch.setattr(
        "minibrain.agent.session_control.observability.find_run_by_request",
        lambda *args, **kwargs: {
            "id": run_id, "question": "问题", "status": "succeeded",
            "answer": "已有答案", "evidence": [], "tool_calls": [], "claims": [],
            "citation_metrics": {}, "context_decisions": [],
            "context_metrics": {}, "confidence_report": {},
        },
    )

    result = previous_result(
        user, session_id="thread", request_id=str(uuid4()), question="问题")

    assert result.answer == "已有答案"
    assert result.run_id == str(run_id)


def test_reusing_request_id_for_different_question_is_conflict(monkeypatch):
    monkeypatch.setattr(
        "minibrain.agent.session_control.observability.find_run_by_request",
        lambda *args, **kwargs: {
            "id": uuid4(), "question": "原问题", "status": "succeeded",
        },
    )
    with pytest.raises(ModuleError) as caught:
        previous_result(
            _user(), session_id="thread", request_id=str(uuid4()), question="新问题")
    assert caught.value.code == "idempotency_conflict"


def test_graph_reuses_idempotent_result_without_calling_model(monkeypatch):
    existing = AnswerResult(answer="缓存结果", run_id="run-1")

    @contextmanager
    def fake_lock(session_id):
        yield

    monkeypatch.setattr(graph, "session_lock", fake_lock)
    monkeypatch.setattr(graph, "previous_result", lambda *args, **kwargs: existing)
    monkeypatch.setattr(
        graph, "_answer_once",
        lambda *args, **kwargs: pytest.fail("幂等命中后不应再次调用 Agent"),
    )

    result = graph.answer(
        _user(), "问题", session_id="thread", request_id=str(uuid4()))

    assert result is existing


def test_postgres_lock_rejects_concurrent_same_thread():
    with session_lock("test-session-lock"):
        with pytest.raises(ModuleError) as caught:
            with session_lock("test-session-lock"):
                pass
    assert caught.value.code == "session_busy"
