"""会话所有权、跨进程串行化与幂等结果恢复。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from uuid import UUID

import psycopg

from ..config import get_config
from ..contracts import Evidence, ModuleError, UserContext
from ..observability import core as observability
from .citations import AnswerClaim, CitationMetrics
from .confidence import ConfidenceReport
from .context import ContextDecision, ContextMetrics
from .types import AnswerResult, ToolCallTrace


def valid_chat_id(user: UserContext, value: str | None) -> bool:
    if not value:
        return False
    owner, separator, thread = value.partition(":")
    if not separator or owner != str(user.user_id):
        return False
    try:
        parsed = UUID(thread)
    except (ValueError, AttributeError):
        return False
    return parsed.version == 4


def normalize_request_id(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ModuleError(
            "request_id 必须是 UUID", code="invalid_request_id", status=400,
        ) from exc


@contextmanager
def session_lock(session_id: str) -> Iterator[None]:
    """用 PostgreSQL session advisory lock 跨进程保护同一个 thread。"""
    connection = psycopg.connect(get_config().database_url, autocommit=True)
    acquired = False
    try:
        row = connection.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
            (f"minibrain:chat:{session_id}",),
        ).fetchone()
        acquired = bool(row[0])
        if not acquired:
            raise ModuleError(
                "该对话上一条问题仍在处理中，请等待完成后再提问。",
                code="session_busy", status=409,
            )
        yield
    finally:
        if acquired:
            connection.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (f"minibrain:chat:{session_id}",),
            )
        connection.close()


def _result_from_run(row: dict) -> AnswerResult:
    metrics = row.get("context_metrics") or None
    confidence = row.get("confidence_report") or None
    return AnswerResult(
        answer=row.get("answer") or "",
        evidence=[Evidence(**item) for item in row.get("evidence", [])],
        trace=[ToolCallTrace(**item) for item in row.get("tool_calls", [])],
        run_id=str(row["id"]),
        claims=[AnswerClaim(**item) for item in row.get("claims", [])],
        citation_metrics=CitationMetrics(**(row.get("citation_metrics") or {})),
        context_decisions=[
            ContextDecision(**item) for item in row.get("context_decisions", [])
        ],
        context_metrics=ContextMetrics(**metrics) if metrics else None,
        confidence_report=ConfidenceReport(**confidence) if confidence else None,
    )


def previous_result(user: UserContext, *, session_id: str, request_id: str,
                    question: str) -> AnswerResult | None:
    row = observability.find_run_by_request(
        user, session_id=session_id, request_id=request_id)
    if row is None:
        return None
    if row["question"] != question:
        raise ModuleError(
            "同一个 request_id 已用于不同问题。",
            code="idempotency_conflict", status=409,
        )
    if row["status"] == "succeeded":
        return _result_from_run(row)
    if row["status"] == "running":
        raise ModuleError(
            "相同请求仍在处理中。", code="request_in_progress", status=409)
    raise ModuleError(
        "相同请求此前已失败；请使用新的 request_id 重试。",
        code="request_previously_failed", status=409,
    )
