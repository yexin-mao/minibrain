"""平台身份层。

职责边界：注册、登录、会话、管理员标记、产出 UserContext。
明确不负责：任何模块内部的 source 权限 —— 那是模块自己判断的事。
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt

from ..config import get_config
from ..contracts import ModuleError, UserContext
from ..db import identity_db


class AuthError(ModuleError):
    def __init__(self, message: str = "用户名或密码错误"):
        super().__init__(message, code="auth_failed", status=401)


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def user_count() -> int:
    with identity_db() as cur:
        cur.execute("SELECT count(*) AS n FROM users")
        return int(cur.fetchone()["n"])


def create_user(username: str, password: str, *, is_admin: bool = False) -> UserContext:
    username = username.strip()
    if len(username) < 2:
        raise ModuleError("用户名至少 2 个字符", code="invalid_username")
    if len(password) < 6:
        raise ModuleError("密码至少 6 个字符", code="invalid_password")

    with identity_db() as cur:
        cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            raise ModuleError("用户名已存在", code="username_taken", status=409)

        cur.execute(
            """
            INSERT INTO users (username, password_hash, is_admin)
            VALUES (%s, %s, %s)
            RETURNING id, username, is_admin
            """,
            (username, _hash(password), is_admin),
        )
        row = cur.fetchone()

    return UserContext(user_id=str(row["id"]), username=row["username"], is_admin=row["is_admin"])


def authenticate(username: str, password: str) -> str:
    """校验密码并签发会话 token。"""
    with identity_db() as cur:
        cur.execute(
            "SELECT id, username, password_hash, is_admin FROM users WHERE username = %s",
            (username.strip(),),
        )
        row = cur.fetchone()

        # 用户不存在时也走一次 hash 校验，避免用响应时间区分"用户不存在"和"密码错"。
        if row is None:
            _verify(password, "$2b$12$" + "x" * 53)
            raise AuthError()
        if not _verify(password, row["password_hash"]):
            raise AuthError()

        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(hours=get_config().session_ttl_hours)
        cur.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (%s, %s, %s)",
            (token, row["id"], expires),
        )

    return token


def resolve_session(token: str | None) -> UserContext | None:
    """token → UserContext。这是平台层唯一向模块交付身份的出口。"""
    if not token:
        return None

    with identity_db() as cur:
        cur.execute(
            """
            SELECT u.id, u.username, u.is_admin
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = %s AND s.expires_at > now()
            """,
            (token,),
        )
        row = cur.fetchone()

    if row is None:
        return None
    return UserContext(user_id=str(row["id"]), username=row["username"], is_admin=row["is_admin"])


def list_users() -> list[UserContext]:
    with identity_db() as cur:
        cur.execute("SELECT id, username, is_admin FROM users ORDER BY created_at")
        return [
            UserContext(user_id=str(r["id"]), username=r["username"], is_admin=r["is_admin"])
            for r in cur.fetchall()
        ]


def delete_user(user_id: str) -> None:
    """删用户及其会话。

    注意：不会连带删掉模块里的数据 —— sources.owner_id 是跨 schema 的逻辑外键，
    刻意不建约束（模块的存储不归身份层管）。调用方必须先让各模块清掉自己的数据。
    """
    with identity_db() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))


def revoke_session(token: str | None) -> None:
    if not token:
        return
    with identity_db() as cur:
        cur.execute("DELETE FROM sessions WHERE token = %s", (token,))
