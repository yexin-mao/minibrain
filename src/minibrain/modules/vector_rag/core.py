"""向量检索链路的全部业务逻辑。

这个文件永远不 import fastapi、不接收 Request、不返回 Response。
它只认 UserContext 和普通参数 —— 这是将来能白送 MCP / CLI / 定时任务的前提。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ...config import get_config
from ...contracts import Evidence, ModuleError, NotFound, PermissionDenied, SearchResult, UserContext
from ...db import vector_db
from .chunking import split_text
from .embeddings import embed_query, embed_texts

MODULE_ID = "vector-rag"


# ---------------------------------------------------------------- 权限
#
# 全平台唯一的可见性判定，永远出现在 SQL 的 WHERE 里，不做查完再筛。
# 应用层后过滤是最容易长出越权 bug 的地方：漏一个分支就是数据泄露。

def _visibility_clause(user: UserContext, alias: str = "s") -> tuple[str, list[Any]]:
    if user.is_admin:
        return "TRUE", []
    return f"({alias}.visibility = 'public' OR {alias}.owner_id = %s)", [user.user_id]


def _require_writable_source(cur, user: UserContext, source_id: str) -> dict:
    cur.execute("SELECT * FROM sources WHERE id = %s", (source_id,))
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


# ---------------------------------------------------------------- source

def list_sources(user: UserContext) -> list[dict]:
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""
            SELECT s.id, s.name, s.visibility, s.owner_id, s.created_at,
                   (SELECT count(*) FROM documents d WHERE d.source_id = s.id) AS document_count
            FROM sources s
            WHERE {where}
            ORDER BY s.visibility DESC, s.name
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def ensure_default_source(user: UserContext) -> dict:
    """每个用户有一个默认私有 source，首次访问时自动建。"""
    name = f"user/{user.username}"
    with vector_db() as cur:
        cur.execute("SELECT * FROM sources WHERE owner_id = %s AND name = %s", (user.user_id, name))
        row = cur.fetchone()
        if row:
            return dict(row)
        cur.execute(
            """
            INSERT INTO sources (name, visibility, owner_id)
            VALUES (%s, 'private', %s)
            RETURNING *
            """,
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

    with vector_db() as cur:
        cur.execute("SELECT 1 FROM sources WHERE owner_id = %s AND name = %s", (user.user_id, name))
        if cur.fetchone():
            raise ModuleError("同名 source 已存在", code="duplicate_source", status=409)
        cur.execute(
            "INSERT INTO sources (name, visibility, owner_id) VALUES (%s, %s, %s) RETURNING *",
            (name, visibility, user.user_id),
        )
        return dict(cur.fetchone())


def delete_source(user: UserContext, source_id: str) -> None:
    """删 source 连带其下全部 document 和 chunk（靠 FK cascade）。

    权限判断和写入走同一条路径：能写才能删。
    """
    with vector_db() as cur:
        _require_writable_source(cur, user, source_id)
        cur.execute("DELETE FROM sources WHERE id = %s", (source_id,))


# ---------------------------------------------------------------- 文档

def list_documents(user: UserContext) -> list[dict]:
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""
            SELECT d.id, d.filename, d.status, d.error, d.char_count, d.created_at,
                   s.name AS source_name, s.visibility,
                   (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            WHERE {where}
            ORDER BY d.created_at DESC
            LIMIT 100
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def upload_document(user: UserContext, source_id: str | None, filename: str, text: str) -> str:
    """只登记，不处理。返回 document_id，真正的解析交给 process_document。

    上传接口必须立刻返回：向量化是分钟级的，同步阻塞在生产上会被反代掐断。
    """
    if source_id:
        with vector_db() as cur:
            _require_writable_source(cur, user, source_id)
        target = source_id
    else:
        target = str(ensure_default_source(user)["id"])

    with vector_db() as cur:
        cur.execute(
            """
            INSERT INTO documents (source_id, filename, content, status, char_count)
            VALUES (%s, %s, %s, 'uploaded', %s)
            RETURNING id
            """,
            (target, filename, text, len(text)),
        )
        return str(cur.fetchone()["id"])


def process_document(document_id: str) -> None:
    """后台线程执行。任何失败都必须落到 failed 状态，绝不伪装成 ready。"""
    cfg = get_config()

    try:
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'processing', updated_at = now() WHERE id = %s",
                (document_id,),
            )
            cur.execute("SELECT source_id, content FROM documents WHERE id = %s", (document_id,))
            row = cur.fetchone()
            if row is None:
                raise NotFound("document 不存在")
            source_id = str(row["source_id"])
            text = row["content"]

        pieces = split_text(text, cfg.chunk_size, cfg.chunk_overlap)
        if not pieces:
            raise ModuleError("文档内容为空，没有可索引的文本", code="empty_document")

        vectors = embed_texts(pieces)

        with vector_db() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            cur.executemany(
                """
                INSERT INTO chunks
                  (document_id, source_id, ordinal, content, embedding, embedding_model, embedding_dim)
                VALUES (%s, %s, %s, %s, %s::real[], %s, %s)
                """,
                [
                    (document_id, source_id, i, piece, vec, cfg.embedding_model, len(vec))
                    for i, (piece, vec) in enumerate(zip(pieces, vectors))
                ],
            )
            cur.execute(
                "UPDATE documents SET status = 'ready', error = NULL, updated_at = now() WHERE id = %s",
                (document_id,),
            )

    except Exception as exc:
        detail = exc.message if isinstance(exc, ModuleError) else f"{type(exc).__name__}: {exc}"
        code = exc.code if isinstance(exc, ModuleError) else "internal_error"
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                (json.dumps({"code": code, "message": detail}, ensure_ascii=False), document_id),
            )


# ---------------------------------------------------------------- 检索

def search(user: UserContext, query: str, top_k: int = 5) -> SearchResult:
    """语义召回。

    先用 SQL 把当前用户看得见的 chunk 捞出来，再在内存里算余弦。
    几千条 chunk 是亚毫秒级；等真到几万条再换 pgvector —— 换的只是这一个函数。
    """
    query = query.strip()
    if not query:
        raise ModuleError("查询不能为空", code="empty_query")

    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""
            SELECT c.content, c.ordinal, c.embedding,
                   d.filename, s.name AS source_name
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN sources s ON s.id = c.source_id
            WHERE {where} AND d.status = 'ready'
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return SearchResult(evidence=[], note="没有可检索的内容：当前用户可见范围内还没有处理完成的文档。")

    matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    vector = np.asarray(embed_query(query), dtype=np.float32)

    if matrix.shape[1] != vector.shape[0]:
        raise ModuleError(
            f"库中向量为 {matrix.shape[1]} 维，当前模型输出 {vector.shape[0]} 维。"
            f"换过 embedding 模型的话需要重新索引。",
            code="embedding_dim_mismatch",
            status=500,
        )

    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    vector /= np.linalg.norm(vector) + 1e-10
    scores = matrix @ vector

    top = np.argsort(-scores)[:top_k]
    evidence = [
        Evidence(
            module=MODULE_ID,
            source_name=rows[i]["source_name"],
            location=f"{rows[i]['filename']} #{rows[i]['ordinal']}",
            snippet=rows[i]["content"],
            score=round(float(scores[i]), 4),
        )
        for i in top
    ]
    return SearchResult(evidence=evidence)


# gateway 后台处理的统一入口名。两条链路各自的处理逻辑完全不同，
# 只在这一层对齐名字，不对齐内部模型。
process = process_document
