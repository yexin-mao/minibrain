"""Web 层的会话。守的是**两条 cookie 各管各的**这条边界。

## 项目里有两个"会话"，别混

| cookie | 决定什么 | 泄露的后果 |
|---|---|---|
| `minibrain_session` | **你是谁**（登录 token） | 被盗号 |
| `minibrain_chat` | **这轮对话的历史存在哪个 thread** | 读到别人的对话 |

它们刻意分开，因为语义和生命周期都不同：合用一个的话，退出再登录会拿到
新 token，对话历史就断了；反过来，想开一个新对话就得重新登录。

## 最关键的一条：thread_id 必须服务端生成

LangGraph 的 checkpointer **只认 `thread_id`，它不知道谁该看哪个 thread**。
所以如果 `/ask` 接受客户端传来的会话 id，改一下 cookie 就能读到别人的对话历史。

`app.py` 里的做法是：cookie 里没有就服务端生成 `{user_id}:{uuid4}`，
**永远不把请求里的值当成新 id 来源**。下面的测试钉住这一点。

★ 这些测试不调真模型：`run_agent` 被替换成一个假的，只回显它收到的
  `session_id`。要测的是**路由怎么管 cookie**，不是模型答得对不对——
  混在一起测会又慢又不稳定。
"""

from __future__ import annotations

import re
import uuid

import pytest
from fastapi.testclient import TestClient

from minibrain import identity
from minibrain.agent.types import AnswerResult
from minibrain.contracts import (
    Evidence, RetrievalCandidate, RetrievalStage, RetrievalTrace, SearchResult,
)
from minibrain.scripts_purge import purge_user
from minibrain.web import app as web_app


@pytest.fixture
def client(monkeypatch):
    """把 agent 换成假的：回显收到的 session_id，不调模型。"""
    def fake_agent(user, question, *, session_id=None, request_id=None):
        return AnswerResult(answer=f"session_id={session_id}")

    monkeypatch.setattr(web_app, "run_agent", fake_agent)
    return TestClient(web_app.app)


@pytest.fixture
def logged_in(client):
    name = "web_" + uuid.uuid4().hex[:6]
    user = identity.create_user(name, "pw123456")
    client.post("/login", data={"username": name, "password": "pw123456"},
                follow_redirects=False)
    yield client
    purge_user(user)


def _session_id(response) -> str:
    """从回显的 HTML 里抠出 session_id。"""
    match = re.search(r"session_id=([\w:-]+)", re.sub(r"<[^>]+>", "", response.text))
    return match.group(1) if match else ""


def test_first_question_creates_a_chat_cookie(logged_in):
    assert "minibrain_chat" not in logged_in.cookies
    logged_in.post("/ask", data={"question": "第一问"})
    assert "minibrain_chat" in logged_in.cookies


def test_ask_passes_client_request_id_to_agent(logged_in, monkeypatch):
    captured = {}

    def fake_agent(user, question, *, session_id=None, request_id=None):
        captured.update(session_id=session_id, request_id=request_id)
        return AnswerResult(answer="ok")

    monkeypatch.setattr(web_app, "run_agent", fake_agent)
    request_id = str(uuid.uuid4())
    response = logged_in.post("/ask", data={
        "question": "问题", "request_id": request_id,
    })

    assert response.status_code == 200
    assert captured["request_id"] == request_id
    assert captured["session_id"]


def test_stream_ask_emits_verified_html_and_sets_chat_cookie(logged_in):
    assert "minibrain_chat" not in logged_in.cookies
    response = logged_in.post("/ask/stream", data={
        "question": "第一问", "request_id": "stream-request-1",
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert "event: started" in response.text
    assert '"request_id":"stream-request-1"' in response.text
    assert "event: completed" in response.text
    assert "session_id=" in response.text
    assert "minibrain_chat" in logged_in.cookies


def test_stream_ask_does_not_leak_unhandled_error(logged_in, monkeypatch):
    def broken_agent(*args, **kwargs):
        raise RuntimeError("secret provider detail")

    monkeypatch.setattr(web_app, "run_agent", broken_agent)
    response = logged_in.post("/ask/stream", data={"question": "问题"})

    assert "event: failed" in response.text
    assert "answer_failed" in response.text
    assert "secret provider detail" not in response.text


def test_home_renders_knowledge_lifecycle_controls(logged_in):
    response = logged_in.get("/")
    assert response.status_code == 200
    assert "创建 source" in response.text
    assert "/knowledge/entity/action" in response.text or "还没有文档" in response.text


def test_retrieval_debug_renders_pipeline_and_uses_explain(logged_in, monkeypatch):
    captured = {}
    real_call = web_app.gateway.call

    def fake_call(module, method, user, *args, **kwargs):
        if (module, method) != ("vector-rag", "search"):
            return real_call(module, method, user, *args, **kwargs)
        query = args[0]
        captured.update(module=module, method=method, query=query, **kwargs)
        candidate = RetrievalCandidate(
            rank=1, node_id="n1", source_name="制度",
            location="policy.md #1", snippet="每天不超过 500 元", score=0.91,
        )
        trace = RetrievalTrace(
            query=query, retrieval_query=query, mode=kwargs["mode"], fetch_k=20,
            business_filters={}, reranker=None, mmr_lambda=0.7,
            stages=[RetrievalStage(
                name="Dense 召回", score_kind="向量相似度",
                candidates=[candidate], latency_ms=12.3,
            )],
        )
        return SearchResult(
            evidence=[Evidence("vector-rag", "制度", "policy.md #1", "正文", 0.91)],
            retrieval_trace=trace,
        )

    monkeypatch.setattr(web_app.gateway, "call", fake_call)
    response = logged_in.post("/retrieval-debug", data={
        "query": "住宿标准", "mode": "hybrid", "top_k": "5",
        "candidate_pool": "20", "use_mmr": "true", "mmr_lambda": "0.7",
    })

    assert response.status_code == 200
    assert captured["module"] == "vector-rag"
    assert captured["method"] == "search"
    assert captured["explain"] is True
    assert captured["use_mmr"] is True
    assert "Dense 召回" in response.text
    assert "每天不超过 500 元" in response.text


def test_xlsx_upload_is_auto_routed_from_vector_to_table(logged_in, monkeypatch):
    real_call = web_app.gateway.call
    captured = {}

    def fake_call(module, method, user, *args, **kwargs):
        if (module, method) == ("table-rag", "detect_upload_kind"):
            return "xlsx"
        if (module, method) == ("table-rag", "upload_bytes"):
            captured.update(module=module, method=method, args=args)
            return ["dataset-1"]
        return real_call(module, method, user, *args, **kwargs)

    monkeypatch.setattr(web_app.gateway, "call", fake_call)
    response = logged_in.post(
        "/upload",
        data={"module": "vector-rag", "source_id": "vector-source-id"},
        files={"file": ("renamed.bin", b"fake-xlsx", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert captured["module"] == "table-rag"
    assert captured["method"] == "upload_bytes"
    assert captured["args"][0] is None  # 不把 vector source_id 带进 table schema
    assert captured["args"][1] == "renamed.bin"


def test_runs_page_renders_trace_summary(logged_in, monkeypatch):
    monkeypatch.setattr(web_app.observability, "list_runs", lambda user: [{
        "id": uuid.uuid4(), "started_at": __import__("datetime").datetime.now(),
        "status": "succeeded", "question": "报销标准是什么？",
        "model": "test-model", "latency_ms": 123, "total_tokens": 18,
        "tool_count": 1, "evidence_count": 2,
        "citation_metrics": {"citation_coverage": 1.0},
        "context_metrics": {"context_tokens": 100, "token_budget": 4000},
    }])

    response = logged_in.get("/runs")
    assert response.status_code == 200
    assert "报销标准是什么？" in response.text
    assert "1 / 2" in response.text


def test_run_detail_renders_tools_and_evidence(logged_in, monkeypatch):
    run_id = str(uuid.uuid4())
    monkeypatch.setattr(web_app.observability, "get_run", lambda user, requested: {
        "id": requested, "status": "succeeded", "model": "test-model",
        "started_at": "2026-08-17", "latency_ms": 321,
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        "question": "问题", "answer": "答案", "error": None,
        "claims": [{
            "text": "每天 500 元", "evidence_ids": ["E1.1"],
            "valid_evidence_ids": ["E1.1"], "invalid_evidence_ids": [],
            "ungrounded_numbers": [], "ungrounded_identifiers": [], "grounded": True,
        }],
        "citation_metrics": {
            "citation_coverage": 1.0, "citation_validity": 1.0,
            "grounded_fact_rate": 1.0,
        },
        "context_decisions": [{
            "evidence_id": "E1.1", "candidate_rank": 1, "token_count": 100,
            "selected": True, "reason": "selected", "source_name": "制度",
            "location": "policy.md#1",
        }],
        "context_metrics": {
            "candidate_count": 1, "selected_count": 1,
            "context_tokens": 100, "token_budget": 4000,
            "budget_utilization": 0.025, "dropped_duplicate_count": 0,
            "dropped_budget_count": 0, "dropped_limit_count": 0,
        },
        "tool_calls": [{
            "name": "vector_search", "arguments": "{}",
            "result_preview": "结果", "step": 1,
        }],
        "evidence": [{
            "module": "vector-rag", "source_name": "制度",
            "location": "policy.md#1", "snippet": "证据正文", "score": 0.9,
            "evidence_id": "E1.1",
        }],
    })

    response = logged_in.get(f"/runs/{run_id}")
    assert response.status_code == 200
    assert "vector_search" in response.text
    assert "证据正文" in response.text


def test_feedback_route_saves_rating_for_current_user(logged_in, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        web_app.observability, "save_feedback",
        lambda user, run_id, **values: {
            **captured.setdefault("call", {
                "user": user, "run_id": run_id, **values,
            }),
            "rating": values["rating"],
        },
    )
    run_id = str(uuid.uuid4())
    response = logged_in.post(f"/runs/{run_id}/feedback", data={
        "rating": "-1", "reason": "incorrect", "note": "数字有误",
    })

    assert response.status_code == 200
    assert captured["call"]["run_id"] == run_id
    assert captured["call"]["rating"] == -1
    assert "反馈已保存" in response.text


def test_feedback_export_marks_samples_for_human_review(logged_in, monkeypatch):
    monkeypatch.setattr(
        web_app.observability, "export_regression_samples",
        lambda user, negative_only=True: [{"case_id": "feedback-1"}],
    )
    response = logged_in.get("/feedback/regression.json")

    assert response.status_code == 200
    assert response.json()["requires_human_review"] is True
    assert response.json()["samples"][0]["case_id"] == "feedback-1"
    assert "attachment" in response.headers["content-disposition"]


def test_metrics_page_renders_aggregate_signals(logged_in, monkeypatch):
    monkeypatch.setattr(web_app.observability, "metrics_overview", lambda user, hours: {
        "hours": hours, "total_runs": 10, "succeeded_runs": 8,
        "failed_runs": 2, "running_runs": 0, "no_evidence_runs": 1,
        "confidence_blocked_runs": 1, "confidence_blocked_rate": 0.125,
        "negative_feedback_runs": 2, "success_rate": 0.8,
        "failure_rate": 0.2, "no_evidence_rate": 0.125,
        "latency_p50_ms": 120, "latency_p95_ms": 900,
        "total_tokens": 1234, "avg_evidence_count": 2.5,
        "avg_citation_coverage": 0.75,
        "failure_codes": [{"code": "timeout", "count": 2}],
    })

    response = logged_in.get("/metrics?hours=168")

    assert response.status_code == 200
    assert "最近 168 小时" in response.text
    assert "120 / 900 ms" in response.text
    assert "timeout" in response.text


def test_same_chat_id_is_reused_across_turns(logged_in):
    """同一个浏览器连续提问，必须落在同一个 thread 上，否则没有上下文。"""
    first = _session_id(logged_in.post("/ask", data={"question": "第一问"}))
    second = _session_id(logged_in.post("/ask", data={"question": "第二问"}))
    assert first and first == second


def test_new_chat_starts_a_different_thread(logged_in):
    before = _session_id(logged_in.post("/ask", data={"question": "第一问"}))
    logged_in.post("/chat/new")
    assert "minibrain_chat" not in logged_in.cookies
    after = _session_id(logged_in.post("/ask", data={"question": "第一问"}))
    assert after and after != before


def test_logout_also_drops_the_chat_context(logged_in):
    """换个人用同一台电脑，不该看到上一个人的对话。"""
    logged_in.post("/ask", data={"question": "第一问"})
    assert "minibrain_chat" in logged_in.cookies
    logged_in.post("/logout", follow_redirects=False)
    assert "minibrain_chat" not in logged_in.cookies


def test_chat_id_is_namespaced_by_user(logged_in):
    """会话 id 带上 user_id 前缀。

    ★ 这不是权限校验（真正的隔离在于「id 不可猜」+「服务端生成」），
      但它让 checkpoint 表在排查时能一眼看出这条属于谁 ——
      出了越权问题，能不能查得清和能不能防住同样重要。
    """
    chat_id = _session_id(logged_in.post("/ask", data={"question": "第一问"}))
    assert ":" in chat_id
    user_part, _, random_part = chat_id.partition(":")
    assert len(user_part) == 36          # uuid4 带连字符
    assert len(random_part) == 32        # uuid4().hex


def test_client_cannot_choose_another_users_thread_id(logged_in):
    """★★ 这条是安全断言，不是行为断言。

    checkpointer 只认 thread_id，不知道谁该看哪个 thread。
    如果 /ask 把客户端传来的值当成新的会话 id 用，
    那么改一下 cookie 就能读到**别人的对话历史**。

    这里伪造一个别人的 thread_id 塞进 cookie，断言服务端要么忽略它、
    要么至少不让它跨到别的用户名下。

    服务端现在校验 user 前缀与 UUID 形状；不匹配时生成新 thread。
    """
    forged = f"{uuid.uuid4()}:{uuid.uuid4().hex}"      # 伪造成"别人的"会话
    logged_in.cookies.set("minibrain_chat", forged)
    used = _session_id(logged_in.post("/ask", data={"question": "第一问"}))

    assert used != forged
    assert used.partition(":")[0] != forged.partition(":")[0]
    assert len(used.partition(":")[2]) == 32
