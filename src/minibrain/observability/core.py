"""Run Trace 的持久化与权限过滤。"""

from __future__ import annotations

import json

from psycopg.types.json import Jsonb

from ..contracts import ModuleError, NotFound, UserContext
from ..db import observability_db


def start_run(user: UserContext, question: str, *, session_id: str | None,
              model: str, request_id: str | None = None) -> str:
    with observability_db() as cur:
        cur.execute(
            """INSERT INTO runs
                 (user_id, session_id, request_id, question, status, model)
               VALUES (%s, %s, %s, %s, 'running', %s) RETURNING id""",
            (user.user_id, session_id, request_id, question, model),
        )
        return str(cur.fetchone()["id"])


def find_run_by_request(user: UserContext, *, session_id: str,
                        request_id: str) -> dict | None:
    """幂等查询只按当前用户 + 当前 thread，管理员权限也不放宽。"""
    with observability_db() as cur:
        cur.execute(
            """SELECT * FROM runs
               WHERE user_id = %s AND session_id = %s AND request_id = %s""",
            (user.user_id, session_id, request_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    result = dict(row)
    for key in (
        "tool_calls", "evidence", "claims", "citation_metrics",
        "context_decisions", "context_metrics", "error", "confidence_report",
    ):
        if isinstance(result.get(key), str):
            result[key] = json.loads(result[key])
    return result


def finish_success(run_id: str, *, answer: str, latency_ms: int,
                   usage: dict[str, int], tool_calls: list[dict],
                   evidence: list[dict], claims: list[dict],
                   citation_metrics: dict, context_decisions: list[dict],
                   context_metrics: dict, raw_answer: str | None = None,
                   confidence_report: dict | None = None) -> None:
    with observability_db() as cur:
        cur.execute(
            """UPDATE runs
               SET status = 'succeeded', answer = %s, finished_at = now(),
                   latency_ms = %s, input_tokens = %s, output_tokens = %s,
                   total_tokens = %s, tool_calls = %s, evidence = %s,
                   claims = %s, citation_metrics = %s,
                   context_decisions = %s, context_metrics = %s,
                   raw_answer = %s, confidence_report = %s, error = NULL
               WHERE id = %s""",
            (answer, latency_ms, usage.get("input_tokens", 0),
             usage.get("output_tokens", 0), usage.get("total_tokens", 0),
             Jsonb(tool_calls), Jsonb(evidence), Jsonb(claims),
             Jsonb(citation_metrics), Jsonb(context_decisions),
             Jsonb(context_metrics), raw_answer,
             Jsonb(confidence_report or {}), run_id),
        )


def finish_failure(run_id: str, *, error: Exception, latency_ms: int) -> None:
    payload = {
        "code": getattr(error, "code", type(error).__name__),
        "message": getattr(error, "message", str(error)),
    }
    with observability_db() as cur:
        cur.execute(
            """UPDATE runs
               SET status = 'failed', finished_at = now(), latency_ms = %s, error = %s
               WHERE id = %s""",
            (latency_ms, Jsonb(payload), run_id),
        )


def archive_stale_runs(timeout_seconds: int) -> int:
    """将进程中断遗留的 running 归档；成功/失败记录永不改写。"""
    with observability_db() as cur:
        cur.execute(
            """UPDATE runs
               SET status = 'failed', finished_at = now(),
                   latency_ms = greatest(0, extract(epoch FROM (now() - started_at)) * 1000)::bigint,
                   error = %s
               WHERE status = 'running'
                 AND started_at < now() - make_interval(secs => %s)
               RETURNING id""",
            (Jsonb({
                "code": "stale_run_archived",
                "message": "问答进程中断或超过运行时限，已由恢复任务归档",
            }), timeout_seconds),
        )
        return len(cur.fetchall())


def _visibility(user: UserContext, *, column: str = "user_id") -> tuple[str, list[object]]:
    if user.is_admin:
        return "TRUE", []
    return f"{column} = %s", [user.user_id]


def list_runs(user: UserContext, *, limit: int = 50) -> list[dict]:
    where, params = _visibility(user)
    safe_limit = max(1, min(limit, 200))
    with observability_db() as cur:
        cur.execute(
            f"""SELECT id, user_id, session_id, question, status, model,
                       started_at, finished_at, latency_ms,
                       input_tokens, output_tokens, total_tokens,
                       jsonb_array_length(tool_calls) AS tool_count,
                       jsonb_array_length(evidence) AS evidence_count,
                       citation_metrics,
                       context_metrics,
                       confidence_report,
                       error
                FROM runs WHERE {where}
                ORDER BY started_at DESC LIMIT %s""",
            [*params, safe_limit],
        )
        return [dict(row) for row in cur.fetchall()]


def get_run(user: UserContext, run_id: str) -> dict:
    where, params = _visibility(user)
    with observability_db() as cur:
        cur.execute(
            f"SELECT * FROM runs WHERE id = %s AND {where}", [run_id, *params])
        row = cur.fetchone()
    if row is None:
        raise NotFound("run 不存在")
    result = dict(row)
    # psycopg 正常会解码 jsonb；这一层兼容被代理返回字符串的环境。
    for key in (
        "tool_calls", "evidence", "claims", "citation_metrics",
        "context_decisions", "context_metrics", "error",
        "confidence_report",
    ):
        if isinstance(result.get(key), str):
            result[key] = json.loads(result[key])
    return result


FEEDBACK_REASONS = {
    "incorrect", "missing_evidence", "bad_citation",
    "irrelevant", "too_verbose", "other",
}


def save_feedback(user: UserContext, run_id: str, *, rating: int,
                  reason: str | None = None, note: str = "") -> dict:
    """保存当前用户对自己一次成功问答的反馈；重复提交覆盖旧值。"""
    if rating not in {-1, 1}:
        raise ModuleError("rating 只能是 -1 或 1", code="invalid_feedback")
    normalized_reason = reason or None
    if normalized_reason not in FEEDBACK_REASONS | {None}:
        raise ModuleError("未知的反馈原因", code="invalid_feedback")
    clean_note = note.strip()
    if len(clean_note) > 1000:
        raise ModuleError("反馈说明不能超过 1000 字", code="invalid_feedback")
    if rating == 1:
        normalized_reason = None

    with observability_db() as cur:
        # 即使管理员能查看所有 run，也不能替其他用户制造反馈样本。
        cur.execute(
            "SELECT status FROM runs WHERE id = %s AND user_id = %s",
            (run_id, user.user_id),
        )
        run = cur.fetchone()
        if run is None:
            raise NotFound("run 不存在")
        if run["status"] != "succeeded":
            raise ModuleError("只能评价成功完成的回答", code="invalid_feedback")
        cur.execute(
            """INSERT INTO run_feedback (run_id, user_id, rating, reason, note)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (run_id, user_id) DO UPDATE
               SET rating = excluded.rating, reason = excluded.reason,
                   note = excluded.note, updated_at = now()
               RETURNING id, run_id, rating, reason, note, updated_at""",
            (run_id, user.user_id, rating, normalized_reason, clean_note),
        )
        return dict(cur.fetchone())


def get_feedback(user: UserContext, run_id: str) -> dict | None:
    """读取当前用户自己的反馈；run 可见性不能放宽反馈所有权。"""
    with observability_db() as cur:
        cur.execute(
            """SELECT id, run_id, rating, reason, note, updated_at
               FROM run_feedback WHERE run_id = %s AND user_id = %s""",
            (run_id, user.user_id),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def export_regression_samples(user: UserContext, *, negative_only: bool = True,
                              limit: int = 500) -> list[dict]:
    """把反馈运行投影成可审查的评测候选；默认只导出差评。"""
    where, params = _visibility(user, column="r.user_id")
    rating_clause = "AND f.rating = -1" if negative_only else ""
    safe_limit = max(1, min(limit, 2000))
    with observability_db() as cur:
        cur.execute(
            f"""SELECT r.id AS run_id, r.question, r.answer, r.raw_answer, r.model,
                       r.evidence, r.claims, r.citation_metrics,
                       r.confidence_report,
                       f.rating, f.reason, f.note, f.updated_at
                FROM run_feedback f
                JOIN runs r ON r.id = f.run_id
                WHERE {where} {rating_clause}
                ORDER BY f.updated_at DESC LIMIT %s""",
            [*params, safe_limit],
        )
        rows = cur.fetchall()
    return [{
        "case_id": f"feedback-{row['run_id']}",
        "question": row["question"],
        "actual_answer": row["answer"],
        "expected_answer": None,
        "review": {
            "rating": row["rating"], "reason": row["reason"],
            "note": row["note"], "reviewed_at": row["updated_at"].isoformat(),
        },
        "run": {
            "id": str(row["run_id"]), "model": row["model"],
            "raw_answer": row["raw_answer"],
            "evidence": row["evidence"], "claims": row["claims"],
            "citation_metrics": row["citation_metrics"],
            "confidence_report": row["confidence_report"],
        },
    } for row in rows]


def metrics_overview(user: UserContext, *, hours: int = 24) -> dict:
    """聚合真实 run trace；普通用户只看自己，管理员看全平台。"""
    safe_hours = max(1, min(hours, 24 * 90))
    where, params = _visibility(user, column="r.user_id")
    with observability_db() as cur:
        cur.execute(
            f"""SELECT
                  count(*) AS total_runs,
                  count(*) FILTER (WHERE r.status = 'succeeded') AS succeeded_runs,
                  count(*) FILTER (WHERE r.status = 'failed') AS failed_runs,
                  count(*) FILTER (WHERE r.status = 'running') AS running_runs,
                  count(*) FILTER (
                    WHERE r.status = 'succeeded'
                      AND jsonb_array_length(r.evidence) = 0
                  ) AS no_evidence_runs,
                  count(*) FILTER (
                    WHERE r.status = 'succeeded'
                      AND r.confidence_report @> '{{"blocked": true}}'::jsonb
                  ) AS confidence_blocked_runs,
                  count(*) FILTER (WHERE f.rating = -1) AS negative_feedback_runs,
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY r.latency_ms)
                    FILTER (WHERE r.latency_ms IS NOT NULL) AS latency_p50_ms,
                  percentile_cont(0.95) WITHIN GROUP (ORDER BY r.latency_ms)
                    FILTER (WHERE r.latency_ms IS NOT NULL) AS latency_p95_ms,
                  coalesce(sum(r.total_tokens), 0) AS total_tokens,
                  avg(jsonb_array_length(r.evidence))
                    FILTER (WHERE r.status = 'succeeded') AS avg_evidence_count,
                  avg((r.citation_metrics->>'citation_coverage')::double precision)
                    FILTER (WHERE r.citation_metrics ? 'citation_coverage'
                            AND r.citation_metrics->>'citation_coverage' <> 'null')
                    AS avg_citation_coverage
                FROM runs r
                LEFT JOIN run_feedback f
                  ON f.run_id = r.id AND f.user_id = r.user_id
                WHERE {where}
                  AND r.started_at >= now() - make_interval(hours => %s)""",
            [*params, safe_hours],
        )
        row = dict(cur.fetchone())
        cur.execute(
            f"""SELECT coalesce(r.error->>'code', 'unknown') AS code,
                       count(*) AS count
                FROM runs r
                WHERE {where} AND r.status = 'failed'
                  AND r.started_at >= now() - make_interval(hours => %s)
                GROUP BY code ORDER BY count DESC, code LIMIT 8""",
            [*params, safe_hours],
        )
        failures = [dict(item) for item in cur.fetchall()]

    total = int(row["total_runs"])
    succeeded = int(row["succeeded_runs"])
    failed = int(row["failed_runs"])
    row.update({
        "hours": safe_hours,
        "total_runs": total,
        "succeeded_runs": succeeded,
        "failed_runs": failed,
        "running_runs": int(row["running_runs"]),
        "no_evidence_runs": int(row["no_evidence_runs"]),
        "confidence_blocked_runs": int(row["confidence_blocked_runs"]),
        "negative_feedback_runs": int(row["negative_feedback_runs"]),
        "success_rate": succeeded / total if total else None,
        "failure_rate": failed / total if total else None,
        "no_evidence_rate": int(row["no_evidence_runs"]) / succeeded
        if succeeded else None,
        "confidence_blocked_rate": int(row["confidence_blocked_runs"]) / succeeded
        if succeeded else None,
        "latency_p50_ms": round(row["latency_p50_ms"])
        if row["latency_p50_ms"] is not None else None,
        "latency_p95_ms": round(row["latency_p95_ms"])
        if row["latency_p95_ms"] is not None else None,
        "avg_evidence_count": float(row["avg_evidence_count"])
        if row["avg_evidence_count"] is not None else None,
        "avg_citation_coverage": float(row["avg_citation_coverage"])
        if row["avg_citation_coverage"] is not None else None,
        "failure_codes": failures,
    })
    return row


def delete_user_runs(user_id: str) -> int:
    """删除一个用户的运行记录；供账号清理流程显式调用。"""
    with observability_db() as cur:
        cur.execute("DELETE FROM runs WHERE user_id = %s", (user_id,))
        return cur.rowcount
