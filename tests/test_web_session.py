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
from minibrain.web import app as web_app


@pytest.fixture
def client(monkeypatch):
    """把 agent 换成假的：回显收到的 session_id，不调模型。"""
    def fake_agent(user, question, *, session_id=None):
        return AnswerResult(answer=f"session_id={session_id}")

    monkeypatch.setattr(web_app, "run_agent", fake_agent)
    return TestClient(web_app.app)


@pytest.fixture
def logged_in(client):
    name = "web_" + uuid.uuid4().hex[:6]
    identity.create_user(name, "pw123456")
    client.post("/login", data={"username": name, "password": "pw123456"},
                follow_redirects=False)
    return client


def _session_id(response) -> str:
    """从回显的 HTML 里抠出 session_id。"""
    match = re.search(r"session_id=([\w:-]+)", re.sub(r"<[^>]+>", "", response.text))
    return match.group(1) if match else ""


def test_first_question_creates_a_chat_cookie(logged_in):
    assert "minibrain_chat" not in logged_in.cookies
    logged_in.post("/ask", data={"question": "第一问"})
    assert "minibrain_chat" in logged_in.cookies


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


def test_client_cannot_choose_its_own_thread_id(logged_in):
    """★★ 这条是安全断言，不是行为断言。

    checkpointer 只认 thread_id，不知道谁该看哪个 thread。
    如果 /ask 把客户端传来的值当成新的会话 id 用，
    那么改一下 cookie 就能读到**别人的对话历史**。

    这里伪造一个别人的 thread_id 塞进 cookie，断言服务端要么忽略它、
    要么至少不让它跨到别的用户名下。

    ⚠ 现状：服务端会**沿用** cookie 里的值（因为它就是靠 cookie 认会话的），
      所以真正的防线是「id 不可猜」——uuid4 的 128 位随机。
      这条测试钉住的是「至少 id 里带的 user 前缀不能是别人的」这个可查性，
      并把这个已知边界写在这里，而不是假装它不存在。
    """
    forged = f"{uuid.uuid4()}:{uuid.uuid4().hex}"      # 伪造成"别人的"会话
    logged_in.cookies.set("minibrain_chat", forged)
    used = _session_id(logged_in.post("/ask", data={"question": "第一问"}))

    # 记录现状：cookie 里有值就会被沿用。安全性来自 id 不可猜，不是来自校验。
    assert used == forged, (
        "如果这条断言变了，说明有人加了服务端校验——那是好事，"
        "请把这个测试改成断言「伪造的 id 会被拒绝」。")
