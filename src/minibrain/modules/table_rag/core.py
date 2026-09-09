"""结构化表格链路的全部业务逻辑。

和向量链路刻意不共享任何内部模型：这里没有 chunk、没有 embedding、没有相似度。
一张报表切碎再向量召回回来算不出正确的合计 —— 这就是两条链路不能合并的理由。

同样：不 import fastapi，只认 UserContext。
"""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
from typing import Any

from psycopg import sql as pgsql

from ...config import get_config
from ...contracts import Evidence, ModuleError, NotFound, PermissionDenied, SearchResult, UserContext
from ...db import table_db, table_readonly_db
from .xlsx import extract_sheet, inspect_workbook, is_xlsx

MODULE_ID = "table-rag"

_MAX_IDENT = 63


# ---------------------------------------------------------------- 权限
# 与向量模块同形但独立实现：两条链路的权限模型允许各自演化，不强行抽象。

def _visibility_clause(user: UserContext, alias: str = "s") -> tuple[str, list[Any]]:
    if user.is_admin:
        return "TRUE", []
    return f"({alias}.visibility = 'public' OR {alias}.owner_id = %s)", [user.user_id]


def _require_writable_source(cur, user: UserContext, source_id: str) -> dict:
    cur.execute("SELECT * FROM sources WHERE id = %s FOR UPDATE", (source_id,))
    source = cur.fetchone()
    if source is None:
        raise NotFound("source 不存在")
    if user.is_admin:
        return source
    if str(source["owner_id"]) != user.user_id:
        raise PermissionDenied("只能写入自己的 source")
    if source["visibility"] == "public":
        raise PermissionDenied("public source 只有管理员可写")
    return source


def _require_writable_dataset(cur, user: UserContext, dataset_id: str) -> dict:
    cur.execute(
        """SELECT d.*, s.name AS source_name, s.visibility, s.owner_id
           FROM datasets d JOIN sources s ON s.id = d.source_id
           WHERE d.id = %s FOR UPDATE OF d, s""",
        (dataset_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise NotFound("dataset 不存在")
    if not user.is_admin:
        if str(row["owner_id"]) != user.user_id:
            raise PermissionDenied("只能管理自己的数据集")
        if row["visibility"] == "public":
            raise PermissionDenied("public source 只有管理员可写")
    return dict(row)


def list_sources(user: UserContext) -> list[dict]:
    where, params = _visibility_clause(user)
    with table_db() as cur:
        cur.execute(
            f"""
            SELECT s.id, s.name, s.visibility, s.owner_id, s.created_at,
                   (SELECT count(*) FROM datasets d WHERE d.source_id = s.id) AS dataset_count
            FROM sources s
            WHERE {where}
            ORDER BY s.visibility DESC, s.name
            """,
            params,
        )
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["can_manage"] = user.is_admin or (
            str(row["owner_id"]) == user.user_id and row["visibility"] == "private")
    return rows


def ensure_default_source(user: UserContext) -> dict:
    name = f"user/{user.username}"
    with table_db() as cur:
        cur.execute("SELECT * FROM sources WHERE owner_id = %s AND name = %s", (user.user_id, name))
        row = cur.fetchone()
        if row:
            return dict(row)
        cur.execute(
            "INSERT INTO sources (name, visibility, owner_id) VALUES (%s, 'private', %s) RETURNING *",
            (name, user.user_id),
        )
        return dict(cur.fetchone())


def create_source(user: UserContext, name: str, visibility: str = "private") -> dict:
    name = name.strip()
    if not name:
        raise ModuleError("source 名称不能为空", code="invalid_name")
    if visibility not in ("private", "public"):
        raise ModuleError("visibility 只能是 private 或 public", code="invalid_visibility")
    if visibility == "public" and not user.is_admin:
        raise PermissionDenied("只有管理员可以创建 public source")
    with table_db() as cur:
        cur.execute("SELECT 1 FROM sources WHERE owner_id = %s AND name = %s",
                    (user.user_id, name))
        if cur.fetchone():
            raise ModuleError("同名 source 已存在", code="duplicate_source", status=409)
        cur.execute(
            "INSERT INTO sources (name, visibility, owner_id) VALUES (%s, %s, %s) RETURNING *",
            (name, visibility, user.user_id),
        )
        return dict(cur.fetchone())


def delete_source(user: UserContext, source_id: str) -> None:
    """删 source。登记表靠 FK cascade，但物理表必须显式 DROP —— 它们不在外键图里。"""
    with table_db() as cur:
        _require_writable_source(cur, user, source_id)
        cur.execute(
            "SELECT 1 FROM datasets WHERE source_id = %s AND status = 'processing' LIMIT 1",
            (source_id,),
        )
        if cur.fetchone():
            raise ModuleError("source 中有正在处理的数据集，请稍后再删",
                              code="source_busy", status=409)
        cur.execute("SELECT table_name FROM datasets WHERE source_id = %s", (source_id,))
        for row in cur.fetchall():
            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(pgsql.Identifier(row["table_name"])))
        cur.execute("DELETE FROM sources WHERE id = %s", (source_id,))


# ---------------------------------------------------------------- 入库

def _normalize_column(raw: str, index: int, seen: set[str]) -> str:
    """列名保留原文（含中文），只做去引号、去换行、截断和去重。

    保留原文是为了让 LLM 写出可读的 SQL；代价是它必须给标识符加双引号，
    这一点在 system prompt 里明确要求。
    """
    name = raw.replace('"', "").replace("\n", " ").strip()
    if not name:
        name = f"col_{index}"
    name = name.encode("utf-8")[:_MAX_IDENT].decode("utf-8", errors="ignore")

    candidate, suffix = name, 2
    while candidate in seen:
        candidate = f"{name}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _looks_numeric(values: list[str]) -> bool:
    seen_any = False
    for value in values:
        text = value.strip().replace(",", "")
        if text == "":
            continue
        # 前导零通常表示编号而不是数值；转 numeric 会把 00123 静默变成 123。
        unsigned = text.lstrip("+-")
        if len(unsigned) > 1 and unsigned.startswith("0") and not unsigned.startswith("0."):
            return False
        seen_any = True
        try:
            float(text)
        except ValueError:
            return False
    return seen_any


def _looks_boolean(values: list[str]) -> bool:
    present = [value.strip().lower() for value in values if value.strip()]
    return bool(present) and all(value in {"true", "false"} for value in present)


def _looks_iso_date(values: list[str]) -> str | None:
    import datetime as dt

    present = [value.strip() for value in values if value.strip()]
    if not present:
        return None
    has_time = False
    for value in present:
        try:
            dt.date.fromisoformat(value)
            continue
        except ValueError:
            pass
        try:
            dt.datetime.fromisoformat(value)
            has_time = True
        except ValueError:
            return None
    return "timestamp" if has_time else "date"


def _infer_type(values: list[str]) -> str:
    if _looks_numeric(values):
        return "numeric"
    if _looks_boolean(values):
        return "boolean"
    return _looks_iso_date(values) or "text"


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ModuleError("无法识别文件编码，请另存为 UTF-8 CSV", code="bad_encoding")


def upload_csv(user: UserContext, source_id: str | None, filename: str, raw: bytes) -> str:
    """只登记，不解析。物理表名由模块生成，绝不让用户输入进标识符。"""
    if len(raw) > get_config().upload_max_bytes:
        raise ModuleError("CSV 超过上传大小上限", code="file_too_large", status=413)
    if source_id:
        with table_db() as cur:
            _require_writable_source(cur, user, source_id)
        target = source_id
    else:
        target = str(ensure_default_source(user)["id"])

    # 编码识别放在上传期：坏编码要立刻报错，不要拖到后台变成一条 failed 记录。
    text = _decode(raw)
    table_name = f"t_{secrets.token_hex(4)}"

    with table_db() as cur:
        cur.execute(
            """
            INSERT INTO datasets (source_id, filename, content, table_name, status)
            VALUES (%s, %s, %s, %s, 'uploaded')
            RETURNING id
            """,
            (target, filename, text, table_name),
        )
        return str(cur.fetchone()["id"])


def detect_upload_kind(user: UserContext, filename: str, raw: bytes) -> str:
    """供 Web 做不可歧义的 XLSX 自动路由；user 参数由 gateway 契约统一注入。"""
    del user, filename
    return "xlsx" if is_xlsx(raw) else "other"


def upload_xlsx(user: UserContext, source_id: str | None,
                filename: str, raw: bytes) -> list[str]:
    """登记一份工作簿和每个可见 Sheet；单元格解析仍由 worker 异步完成。"""
    if len(raw) > get_config().upload_max_bytes:
        raise ModuleError("XLSX 超过上传大小上限", code="file_too_large", status=413)
    info = inspect_workbook(raw, filename)
    if source_id:
        with table_db() as cur:
            _require_writable_source(cur, user, source_id)
        target = source_id
    else:
        target = str(ensure_default_source(user)["id"])

    with table_db() as cur:
        cur.execute(
            "INSERT INTO workbooks (source_id, filename, content) "
            "VALUES (%s, %s, %s) RETURNING id",
            (target, filename, raw),
        )
        workbook_id = str(cur.fetchone()["id"])
        ids = []
        for sheet_name in info.visible_sheets:
            cur.execute(
                """INSERT INTO datasets
                     (source_id, filename, content, table_name, status,
                      workbook_id, sheet_name, parsed_as)
                   VALUES (%s, %s, '', %s, 'uploaded', %s, %s, 'xlsx')
                   RETURNING id""",
                (target, filename, f"t_{secrets.token_hex(4)}", workbook_id, sheet_name),
            )
            ids.append(str(cur.fetchone()["id"]))
    return ids


def upload_bytes(user: UserContext, source_id: str | None,
                 filename: str, raw: bytes) -> str | list[str]:
    """表格统一上传入口。XLSX 内容识别优先，其余按 CSV 处理。"""
    if is_xlsx(raw):
        return upload_xlsx(user, source_id, filename, raw)
    return upload_csv(user, source_id, filename, raw)


def _claim_specific(dataset_id: str) -> bool:
    lease = get_config().ingest_lease_seconds
    with table_db() as cur:
        cur.execute(
            """WITH target AS (
                 SELECT d.id FROM datasets d JOIN sources s ON s.id = d.source_id
                 WHERE d.id = %s AND (
                      (d.status = 'uploaded' AND d.available_at <= now())
                      OR (d.status = 'processing'
                          AND (d.lease_until IS NULL OR d.lease_until < now()))
                 )
                 FOR UPDATE OF d, s
               )
               UPDATE datasets d
               SET status = 'processing', attempt_count = attempt_count + 1,
                   processing_started_at = now(),
                   lease_until = now() + make_interval(secs => %s),
                   updated_at = now()
               FROM target t WHERE d.id = t.id
               RETURNING d.id""",
            (dataset_id, lease),
        )
        return cur.fetchone() is not None


def claim_next() -> str | None:
    """原子领取一个待处理或 lease 已过期的数据集。"""
    lease = get_config().ingest_lease_seconds
    with table_db() as cur:
        cur.execute(
            """WITH candidate AS (
                 SELECT d.id FROM datasets d JOIN sources s ON s.id = d.source_id
                 WHERE (d.status = 'uploaded' AND d.available_at <= now())
                    OR (d.status = 'processing'
                        AND (d.lease_until IS NULL OR d.lease_until < now()))
                 ORDER BY d.available_at, d.created_at
                 FOR UPDATE OF d, s SKIP LOCKED
                 LIMIT 1
               )
               UPDATE datasets d
               SET status = 'processing', attempt_count = d.attempt_count + 1,
                   processing_started_at = now(),
                   lease_until = now() + make_interval(secs => %s),
                   updated_at = now()
               FROM candidate c WHERE d.id = c.id
               RETURNING d.id""",
            (lease,),
        )
        row = cur.fetchone()
        return str(row["id"]) if row else None


def renew_lease(dataset_id: str) -> bool:
    """处理中的 worker 心跳；只延长仍由 worker 处理的任务。"""
    lease = get_config().ingest_lease_seconds
    with table_db() as cur:
        cur.execute(
            """UPDATE datasets
               SET lease_until = now() + make_interval(secs => %s), updated_at = now()
               WHERE id = %s AND status = 'processing' RETURNING id""",
            (lease, dataset_id),
        )
        return cur.fetchone() is not None


def _record_failure(dataset_id: str, exc: Exception, attempt_count: int) -> None:
    cfg = get_config()
    code = getattr(exc, "code", type(exc).__name__)
    detail = getattr(exc, "message", str(exc))
    retryable = not isinstance(exc, ModuleError)
    payload = json.dumps({
        "code": code, "message": detail, "retryable": retryable,
        "attempt": attempt_count,
    }, ensure_ascii=False)
    with table_db() as cur:
        if retryable and attempt_count < cfg.ingest_max_attempts:
            delay = cfg.ingest_retry_base_seconds * (2 ** max(attempt_count - 1, 0))
            cur.execute(
                """UPDATE datasets
                   SET status = 'uploaded', error = %s,
                       available_at = now() + make_interval(secs => %s),
                       lease_until = NULL, updated_at = now()
                   WHERE id = %s AND status = 'processing'""",
                (payload, delay, dataset_id),
            )
        else:
            terminal = "dead_letter" if retryable else "failed"
            cur.execute(
                """UPDATE datasets
                   SET status = %s, error = %s, lease_until = NULL,
                       finished_at = now(), updated_at = now()
                   WHERE id = %s AND status = 'processing'""",
                (terminal, payload, dataset_id),
            )


def process_dataset(dataset_id: str, *, claimed: bool = False) -> None:
    """后台线程执行：解析 CSV/XLSX Sheet → 建物理表 → 灌行。"""
    if not claimed and not _claim_specific(dataset_id):
        return
    attempt_count = 1
    try:
        with table_db() as cur:
            cur.execute(
                """SELECT d.table_name, d.content, d.attempt_count, d.parsed_as,
                          d.sheet_name, d.filename, w.content AS workbook_content
                   FROM datasets d
                   LEFT JOIN workbooks w ON w.id = d.workbook_id
                   WHERE d.id = %s AND d.status = 'processing'""", (dataset_id,))
            row = cur.fetchone()
            if row is None:
                return
            attempt_count = int(row["attempt_count"])
            table_name = row["table_name"]
            if row["parsed_as"] == "xlsx":
                if row["workbook_content"] is None or not row["sheet_name"]:
                    raise ModuleError("XLSX 原文或 Sheet 信息缺失", code="xlsx_storage_missing")
                rows = extract_sheet(
                    bytes(row["workbook_content"]), row["filename"], row["sheet_name"])
            else:
                reader = csv.reader(io.StringIO(row["content"]))
                rows = [r for r in reader if any(cell.strip() for cell in r)]
        if len(rows) < 2:
            raise ModuleError("表格至少需要表头和一行数据", code="empty_table")

        header, body = rows[0], rows[1:]
        seen: set[str] = set()
        columns = [_normalize_column(name, i, seen) for i, name in enumerate(header)]

        width = len(columns)
        body = [r[:width] + [""] * (width - len(r)) for r in body]

        types = [_infer_type([r[i] for r in body]) for i in range(width)]

        ident = pgsql.Identifier(table_name)
        column_defs = pgsql.SQL(", ").join(
            pgsql.SQL("{} {}").format(pgsql.Identifier(col), pgsql.SQL(typ))
            for col, typ in zip(columns, types)
        )

        with table_db() as cur:
            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(ident))
            cur.execute(pgsql.SQL("CREATE TABLE {} ({})").format(ident, column_defs))

            placeholders = pgsql.SQL(", ").join(pgsql.Placeholder() * width)
            insert = pgsql.SQL("INSERT INTO {} VALUES ({})").format(ident, placeholders)
            cur.executemany(
                insert,
                [
                    [
                        (None if cell.strip() == "" else
                         (cell.strip().replace(",", "") if typ == "numeric" else
                          cell.strip().lower() if typ == "boolean" else cell.strip()))
                        for cell, typ in zip(r, types)
                    ]
                    for r in body
                ],
            )

            cur.execute(
                """
                UPDATE datasets
                SET columns = %s, row_count = %s, status = 'ready', error = NULL,
                    lease_until = NULL, finished_at = now(), updated_at = now()
                WHERE id = %s AND status = 'processing'
                """,
                (
                    json.dumps(
                        [{"name": c, "type": t} for c, t in zip(columns, types)], ensure_ascii=False
                    ),
                    len(body),
                    dataset_id,
                ),
            )

    except Exception as exc:
        _record_failure(dataset_id, exc, attempt_count)


def list_datasets(user: UserContext) -> list[dict]:
    where, params = _visibility_clause(user)
    with table_db() as cur:
        cur.execute(
            f"""
            SELECT d.id, d.filename, d.table_name, d.columns, d.row_count,
                   d.parsed_as, d.sheet_name, d.workbook_id,
                   d.status, d.error, d.created_at, d.updated_at,
                   d.attempt_count, d.available_at, d.lease_until,
                   d.processing_started_at, d.finished_at,
                   s.name AS source_name, s.visibility, s.owner_id
            FROM datasets d
            JOIN sources s ON s.id = d.source_id
            WHERE {where}
            ORDER BY d.created_at DESC
            LIMIT 100
            """,
            params,
        )
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["can_manage"] = user.is_admin or (
            str(row["owner_id"]) == user.user_id and row["visibility"] == "private")
    return rows


def delete_dataset(user: UserContext, dataset_id: str) -> None:
    """删除登记记录及其真实物理表。"""
    with table_db() as cur:
        row = _require_writable_dataset(cur, user, dataset_id)
        if row["status"] == "processing":
            raise ModuleError("数据集正在处理，请稍后再删", code="dataset_busy", status=409)
        cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(
            pgsql.Identifier(row["table_name"])))
        cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))
        if row.get("workbook_id"):
            cur.execute(
                "DELETE FROM workbooks w WHERE w.id = %s "
                "AND NOT EXISTS (SELECT 1 FROM datasets d WHERE d.workbook_id = w.id)",
                (row["workbook_id"],),
            )


def retry_dataset(user: UserContext, dataset_id: str) -> str:
    """把 failed/dead_letter 数据集重新排入处理。"""
    with table_db() as cur:
        row = _require_writable_dataset(cur, user, dataset_id)
        if row["status"] not in {"failed", "dead_letter"}:
            raise ModuleError("只有 failed/dead_letter 数据集可以重试",
                              code="invalid_dataset_state", status=409)
        cur.execute(
            "UPDATE datasets SET status = 'uploaded', error = NULL, attempt_count = 0, "
            "available_at = now(), lease_until = NULL, processing_started_at = NULL, "
            "finished_at = NULL, updated_at = now() "
            "WHERE id = %s", (dataset_id,))
    return dataset_id


def reprocess_dataset(user: UserContext, dataset_id: str) -> str:
    """将 ready 数据集重新解析；旧物理表在新事务提交前仍保持完整。"""
    with table_db() as cur:
        row = _require_writable_dataset(cur, user, dataset_id)
        if row["status"] != "ready":
            raise ModuleError("只有 ready 数据集可以重新处理",
                              code="invalid_dataset_state", status=409)
        cur.execute(
            "UPDATE datasets SET status = 'uploaded', error = NULL, attempt_count = 0, "
            "available_at = now(), lease_until = NULL, processing_started_at = NULL, "
            "finished_at = NULL, updated_at = now() "
            "WHERE id = %s", (dataset_id,))
    return dataset_id


# ---------------------------------------------------------------- 只读查询
#
# 护栏是四层，任何一层单独都不够：
#   1. 表名白名单 —— 权限的真正落点，用 SQL 先算出当前用户看得见哪些表
#   2. 只允许单条 SELECT / WITH
#   3. 强制外层 LIMIT
#   4. 只读事务 + statement_timeout（连接层面，不依赖上面三层的正确性）

_IDENT_AFTER_FROM = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_CTE_NAME = re.compile(r"\b(?:with|,)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", re.IGNORECASE)


def visible_tables(user: UserContext) -> list[dict]:
    """当前用户可查询的表清单。这是权限进入表格链路的唯一入口。"""
    where, params = _visibility_clause(user)
    with table_db() as cur:
        cur.execute(
            f"""
            SELECT d.table_name, d.filename, d.sheet_name, d.parsed_as,
                   d.columns, d.row_count, s.name AS source_name
            FROM datasets d
            JOIN sources s ON s.id = d.source_id
            WHERE {where} AND d.status = 'ready'
            ORDER BY d.created_at DESC
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


# 低基数文本列的取值一并注入 schema 说明。
#
# 起因：tbl-18「有几笔报销被驳回」，模型写 WHERE "状态" = '驳回'，
# 而表里的实际值是 '已驳回'，返回 0（真值 2）。
# 根因是 describe_schema 只注入列名和类型——那是入库时从 CSV 表头推断出来的元数据，
# 从来没有 SELECT DISTINCT 看过实际取值。模型只能猜枚举值。
#
# 这在 text-to-SQL 领域叫 value linking（schema linking 的一部分），是公认的瓶颈。
# 全量注入取值在大表上不可行，所以要设阈值：
_ENUM_MAX_VALUES = 12    # 不同取值超过这么多就不注入（那是高基数列，列出来没意义还挤占上下文）
_ENUM_MAX_LEN = 40       # 单个取值超过这么长也不注入（那多半是自由文本，不是枚举）
_ENUM_MAX_TABLES = 20    # 表太多时只对前 N 张做，避免 describe_schema 变慢


def _enum_values(cur, table_name: str, column: str) -> list[str] | None:
    """取一列的全部不同取值。高基数或长文本返回 None，表示不该注入。

    多取一条（LIMIT n+1）就能判断"是不是超过阈值"，不必先 count 一遍。
    """
    try:
        cur.execute(
            pgsql.SQL("SELECT DISTINCT {} AS v FROM {} WHERE {} IS NOT NULL LIMIT %s").format(
                pgsql.Identifier(column), pgsql.Identifier(table_name), pgsql.Identifier(column)
            ),
            (_ENUM_MAX_VALUES + 1,),
        )
        values = [str(row["v"]) for row in cur.fetchall()]
    except Exception:                       # noqa: BLE001  表刚被删等情况，不该让整个 prompt 失败
        return None

    if not values or len(values) > _ENUM_MAX_VALUES:
        return None
    if any(len(v) > _ENUM_MAX_LEN for v in values):
        return None
    return sorted(values)


def describe_schema(user: UserContext) -> str:
    """给 LLM 看的表结构说明。只描述它有权查询的表。

    除列名和类型外，还注入**低基数文本列的实际取值**——
    否则模型只能猜枚举值，写出 WHERE "状态" = '驳回' 这种匹配不上的条件。
    """
    tables = visible_tables(user)
    if not tables:
        return "（当前用户可见范围内没有任何已就绪的数据表）"

    lines = []
    with table_db() as cur:
        for index, t in enumerate(tables):
            cols = ", ".join(f'"{c["name"]}" {c["type"]}' for c in t["columns"])
            block = (
                f'表 {t["table_name"]}（来自 {t["filename"]}'
                + (f' / Sheet {t["sheet_name"]}' if t.get("sheet_name") else '')
                + f'，source={t["source_name"]}，'
                f'{t["row_count"]} 行）\n  列：{cols}'
            )

            if index < _ENUM_MAX_TABLES:
                # 压成一行。第一版每列一行，把表格段从 210 撑到 668 字符，
                # 结果模型的注意力又被拉回表格，路由准确率从 95.3% 掉到 88~91%——
                # 信息对等不是一次性达成的，加了一边就得重新平衡。
                enums = []
                for column in t["columns"]:
                    if column["type"] != "text":
                        continue
                    values = _enum_values(cur, t["table_name"], column["name"])
                    if values:
                        enums.append(f'{column["name"]}={"/".join(values)}')
                if enums:
                    block += "\n  取值：" + "；".join(enums)

            lines.append(block)
    return "\n".join(lines)


def run_query(user: UserContext, statement: str) -> dict:
    cfg = get_config()
    text = statement.strip().rstrip(";").strip()

    if not text:
        raise ModuleError("SQL 不能为空", code="empty_sql")
    if ";" in text:
        raise ModuleError("只允许单条语句", code="multiple_statements")
    if not re.match(r"^\s*(select|with)\b", text, re.IGNORECASE):
        raise ModuleError("只允许 SELECT 查询", code="not_a_select")

    allowed = {t["table_name"] for t in visible_tables(user)}
    cte_names = {m.lower() for m in _CTE_NAME.findall(text)}
    referenced = {m.lower() for m in _IDENT_AFTER_FROM.findall(text)}

    unknown = referenced - allowed - cte_names
    if unknown:
        raise PermissionDenied(
            f"查询引用了不可见或不存在的表：{', '.join(sorted(unknown))}。"
            f"当前可查询：{', '.join(sorted(allowed)) or '（无）'}"
        )

    wrapped = f"SELECT * FROM ({text}) AS _guarded LIMIT {cfg.table_query_max_rows}"

    try:
        with table_readonly_db() as cur:
            cur.execute(wrapped)
            rows = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        raise ModuleError(f"SQL 执行失败：{exc}", code="sql_failed") from exc

    columns = list(rows[0].keys()) if rows else []
    return {"sql": text, "columns": columns, "rows": rows, "row_count": len(rows)}


def search(user: UserContext, query: str, top_k: int = 5) -> SearchResult:
    """表格链路没有"语义检索"。这里返回可查询的表清单，供 Agent 决定下一步写什么 SQL。"""
    tables = visible_tables(user)
    if not tables:
        return SearchResult(evidence=[], note="当前用户可见范围内没有已就绪的数据表。")

    evidence = [
        Evidence(
            module=MODULE_ID,
            source_name=t["source_name"],
            location=t["table_name"],
            snippet=f'{t["filename"]}'
                    + (f' / {t["sheet_name"]}' if t.get("sheet_name") else '')
                    + f'：{t["row_count"]} 行，列 '
                    + ", ".join(c["name"] for c in t["columns"]),
        )
        for t in tables[:top_k]
    ]
    return SearchResult(evidence=evidence, note="表格链路请用 table_query 工具写 SQL 查询。")


process = process_dataset
