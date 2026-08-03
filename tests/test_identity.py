"""身份层：登录、会话、错误密码。

身份层的职责边界是"你是谁""是不是管理员"，仅此而已。
模块内部的 source 权限不归它管——那部分在 test_permissions.py。
"""

from __future__ import annotations

import pytest

from minibrain import identity
from minibrain.contracts import ModuleError


def test_authenticate_issues_token(users):
    token = identity.authenticate(f"smoke_a_{users['tag']}", "pw123456")
    assert token


def test_token_resolves_to_user_context(users, alice):
    token = identity.authenticate(f"smoke_a_{users['tag']}", "pw123456")
    resolved = identity.resolve_session(token)
    assert resolved is not None
    assert resolved.user_id == alice.user_id
    assert resolved.is_admin is False


def test_garbage_token_resolves_to_none():
    assert identity.resolve_session("garbage") is None


def test_empty_token_resolves_to_none():
    assert identity.resolve_session(None) is None
    assert identity.resolve_session("") is None


def test_wrong_password_is_rejected(users):
    with pytest.raises(ModuleError):
        identity.authenticate(f"smoke_a_{users['tag']}", "wrong")


def test_unknown_user_is_rejected():
    """不存在的用户和密码错误必须返回同一种错误。

    core.py 里对不存在的用户也会跑一次假 hash 校验，
    为的是不让响应时间泄露"这个用户名存不存在"。
    """
    with pytest.raises(ModuleError):
        identity.authenticate("definitely_not_a_user_xyz", "pw123456")


def test_revoked_session_stops_working(users):
    token = identity.authenticate(f"smoke_a_{users['tag']}", "pw123456")
    assert identity.resolve_session(token) is not None
    identity.revoke_session(token)
    assert identity.resolve_session(token) is None
