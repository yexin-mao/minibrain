"""向量链路的**注册与权限**：source / document 的 CRUD、可见性判定、上传解析。

这个文件永远不 import fastapi、不接收 Request、不返回 Response。
它只认 UserContext 和普通参数 —— 这是将来能白送 MCP / CLI / 定时任务的前提。

## 这里没有检索逻辑，是故意的

切分 / 向量检索 / BM25 / 融合 / 重排全部交给 LlamaIndex，在 `chain.py`。
本文件只管**框架管不了的那部分**：

- **权限**：`_visibility_clause` 是全平台唯一的可见性判定，永远出现在 SQL 的
  WHERE 里，不做查完再筛。框架版把它翻译成 `MetadataFilters`，落点变了、规矩没变。
- **文档注册表**：`documents` 表记录状态和失败原因。LlamaIndex 把节点存在
  自己的 `mod_vector_li.data_nodes` 里，不管"这篇文档处理到哪一步了"。
- **上传解析**：类型识别、PDF 抽取、拒收二进制（`loaders.py`）。

被换下来的手写检索实现在 `minibrain/handwritten/vector_search.py`，
作为评测基线保留。
"""

from __future__ import annotations

import json
from typing import Any

from ...config import get_config
from ...contracts import ModuleError, NotFound, PermissionDenied, UserContext
from ...db import vector_db
from .loaders import UnsupportedFile, load_text
from .metadata import extract_document_metadata

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


def _require_writable_document(cur, user: UserContext, document_id: str) -> dict:
    cur.execute(
        """SELECT d.*, s.name AS source_name, s.visibility, s.owner_id
           FROM documents d JOIN sources s ON s.id = d.source_id
           WHERE d.id = %s FOR UPDATE OF d, s""",
        (document_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise NotFound("document 不存在")
    if not user.is_admin:
        if str(row["owner_id"]) != user.user_id:
            raise PermissionDenied("只能管理自己的文档")
        if row["visibility"] == "public":
            raise PermissionDenied("public source 只有管理员可写")
    return dict(row)


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
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["can_manage"] = user.is_admin or (
            str(row["owner_id"]) == user.user_id and row["visibility"] == "private")
    return rows


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
    from . import chain

    with vector_db() as cur:
        source = _require_writable_source(cur, user, source_id)
        cur.execute(
            "SELECT 1 FROM documents WHERE source_id = %s AND status = 'processing' LIMIT 1",
            (source_id,),
        )
        if cur.fetchone():
            raise ModuleError("source 中有正在处理的文档，请稍后再删",
                              code="source_busy", status=409)
        chain.delete_source_nodes(
            owner_id=str(source["owner_id"]), source_name=str(source["name"]))
        cur.execute("DELETE FROM sources WHERE id = %s", (source_id,))


# ---------------------------------------------------------------- 文档

def list_documents(user: UserContext) -> list[dict]:
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""
            SELECT d.id, d.filename, d.status, d.error, d.char_count,
                   d.content_hash, d.version, d.created_at,
                   d.updated_at, d.attempt_count, d.available_at, d.lease_until,
                   d.processing_started_at, d.finished_at,
                   d.parsed_as, s.name AS source_name, s.visibility, s.owner_id,
                   -- ★ 用缓存列而不是 count(chunks)：主路径的节点存在
                   -- LlamaIndex 自己的表里（mod_vector_li.data_nodes），
                   -- mod_vector.chunks 只有手写版才写。
                   d.chunk_count, w.id AS wiki_page_id, w.status AS wiki_status
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            LEFT JOIN wiki_pages w ON w.document_id = d.id
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


def get_document_revision(user: UserContext, document_id: str, version: int) -> dict:
    """读取用户可见的不可变解析原文版本，供 Wiki 来源追溯。"""
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""SELECT r.document_id, r.version, r.content, r.content_hash,
                       r.parsed_as, r.char_count, r.created_at,
                       d.filename, d.version AS current_version,
                       s.name AS source_name, s.visibility
                FROM document_revisions r
                JOIN documents d ON d.id = r.document_id
                JOIN sources s ON s.id = d.source_id
                WHERE r.document_id = %s AND r.version = %s AND {where}""",
            [document_id, int(version), *params],
        )
        row = cur.fetchone()
    if row is None:
        raise NotFound("原文版本不存在")
    return dict(row)


def delete_document(user: UserContext, document_id: str) -> None:
    """删除注册文档及其 dense/sparse 节点；processing 状态拒绝并发删除。"""
    from . import chain

    with vector_db() as cur:
        row = _require_writable_document(cur, user, document_id)
        if row["status"] == "processing":
            raise ModuleError("文档正在处理，请稍后再删", code="document_busy", status=409)
        chain.delete_document_nodes(
            owner_id=str(row["owner_id"]), document_id=document_id)
        cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def retry_document(user: UserContext, document_id: str) -> str:
    """把失败文档重新排入处理；只改变状态，调用方负责调度 process。"""
    from . import chain

    with vector_db() as cur:
        row = _require_writable_document(cur, user, document_id)
        if row["status"] not in {"failed", "dead_letter"}:
            raise ModuleError("只有 failed/dead_letter 文档可以重试", code="invalid_document_state", status=409)
        if not (row["content"] or "").strip():
            raise ModuleError("文档正文为空，无法重试", code="not_retryable", status=409)
        # 上一次若在 dense 写入后失败，先清掉可能的半成品；document_id 让这一步
        # 不会误删同 source 下的同名文件。
        chain.delete_document_nodes(
            owner_id=str(row["owner_id"]), document_id=document_id)
        cur.execute(
            "UPDATE documents SET status = 'uploaded', error = NULL, chunk_count = 0, "
            "attempt_count = 0, available_at = now(), lease_until = NULL, "
            "processing_started_at = NULL, finished_at = NULL, "
            "updated_at = now() WHERE id = %s",
            (document_id,),
        )
    return document_id


def reindex_document(user: UserContext, document_id: str) -> str:
    """清理旧索引并把 ready 文档重新排入处理。"""
    from . import chain

    with vector_db() as cur:
        row = _require_writable_document(cur, user, document_id)
        if row["status"] != "ready":
            raise ModuleError("只有 ready 文档可以重新索引",
                              code="invalid_document_state", status=409)
        chain.delete_document_nodes(
            owner_id=str(row["owner_id"]), document_id=document_id)
        cur.execute(
            "UPDATE documents SET status = 'uploaded', error = NULL, chunk_count = 0, "
            "attempt_count = 0, available_at = now(), lease_until = NULL, "
            "processing_started_at = NULL, finished_at = NULL, "
            "updated_at = now() WHERE id = %s",
            (document_id,),
        )
    return document_id


def describe_corpus(user: UserContext, *, with_titles: bool = True) -> str:
    """给 LLM 看的文档清单。和 table_rag.describe_schema 对称。

    为什么需要它：system prompt 里表格侧列了表名、列名、行数，文档侧原来只有
    一句"适合非结构化文档"。模型看得见表里有哪些列，却完全不知道文档里写了什么，
    于是凡是两边都有的事实（组长、人数、部门归属）它一律去查表。
    消融实验（scripts/ablate_prompt.py）证实了这个信息不对等是路由失败的根因。

    with_titles=False 只列文件名——实验里这一版**没有效果**，保留是为了让
    "光给文件名不够"这个结论可复现，不是为了给调用方选。
    """
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""
            SELECT d.filename, d.content, s.name AS source_name
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            WHERE {where} AND d.status = 'ready'
            ORDER BY d.created_at
            LIMIT 200
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return "（当前用户可见范围内没有任何已就绪的文档）"

    lines = []
    for row in rows:
        if not with_titles:
            lines.append(f"  {row['filename']}")
            continue
        title = _first_heading(row["content"]) or row["filename"].rsplit(".", 1)[0]
        lines.append(f"  {row['filename']} —— {title}")
    return "\n".join(lines)


def describe_domains(user: UserContext) -> str:
    """给 Router 的 O(source 数) 摘要；source 就是现有的知识域边界。"""
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""SELECT s.name, s.visibility, count(d.id) AS document_count,
                       string_agg(DISTINCT d.parsed_as, ', ' ORDER BY d.parsed_as) AS formats
                FROM sources s
                JOIN documents d ON d.source_id = s.id AND d.status = 'ready'
                WHERE {where}
                GROUP BY s.id, s.name, s.visibility
                ORDER BY s.visibility DESC, s.name""",
            params,
        )
        rows = cur.fetchall()
    if not rows:
        return "（当前用户可见范围内没有任何已就绪的知识域）"
    return "\n".join(
        f"- {row['name']}（{row['visibility']}，{row['document_count']} 篇，"
        f"格式：{row['formats'] or 'text'}）"
        for row in rows
    )


def _first_heading(text: str) -> str | None:
    """取正文里第一个 markdown 标题作为文档主题。没有标题就返回 None。"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def knowledge_version(user: UserContext) -> str:
    """当前用户可见 ready 文档的廉价版本戳，用于权限安全的缓存失效。"""
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""SELECT count(*) AS n,
                       coalesce(max(d.updated_at)::text, '') AS latest,
                       coalesce(sum(d.version), 0) AS versions
                FROM documents d JOIN sources s ON s.id = d.source_id
                WHERE {where} AND d.status = 'ready'""",
            params,
        )
        row = cur.fetchone()
    return f"{row['n']}:{row['latest']}:{row['versions']}"


def upload_bytes(user: UserContext, source_id: str | None,
                 filename: str, raw: bytes) -> str:
    """上传原始字节。**这是 Web 上传该走的入口。**

    ★★ 为什么要有这个函数，而不是让调用方自己 decode：

    原来 web 层是这么干的：

        raw.decode("utf-8", "replace")

    传个 PDF 进去，二进制被强行解码成一大片替换字符，然后**照常切分、
    照常向量化、状态标成 ready**。用户以为传成功了，检索时永远命中不了。

    这违反了本项目最核心的一条规矩——「任何失败都必须落 failed，
    绝不伪装成 ready」。那条规矩写在 process_document 上，
    **但上传这一步在它之前，漏掉了**。

    现在解析放在这里做：解析不了就**先落一条 failed 记录再抛异常**。
    为什么要先落记录：用户在文档列表里得看得见「这篇失败了，原因是什么」。
    直接抛异常而不留痕的话，文件像是从没上传过——排查时无从下手。
    """
    text, how = "", "unknown"
    error: str | None = None
    error_code, error_status = "unsupported_file", 415
    if len(raw) > get_config().upload_max_bytes:
        limit_mb = get_config().upload_max_bytes / (1024 * 1024)
        error = f"{filename} 超过上传上限 {limit_mb:g} MB"
        error_code, error_status = "file_too_large", 413
    else:
        try:
            text, how = load_text(raw, filename)
        except UnsupportedFile as exc:
            error = str(exc)

    # error 列是 jsonb（结构化错误），和 process_document 里的失败写法保持一致：
    # {"code": ..., "message": ...}。前端和排查脚本只认这一种形状。
    payload = (json.dumps({"code": error_code, "message": error},
                          ensure_ascii=False) if error else None)
    doc_id = _insert_document(user, source_id, filename, text,
                              status="failed" if error else "uploaded",
                              error=payload, parsed_as=how)
    if error:
        raise ModuleError(error, code=error_code, status=error_status)
    return doc_id


def upload_document(user: UserContext, source_id: str | None, filename: str, text: str) -> str:
    """已经是纯文本时的入口（评测脚本、CLI 导入走这条）。

    Web 上传请走 `upload_bytes` —— 那条路会做类型识别和解析。
    """
    return _insert_document(user, source_id, filename, text,
                            status="uploaded", error=None, parsed_as="text")


def _insert_document(user: UserContext, source_id: str | None, filename: str,
                     text: str, *, status: str, error: str | None,
                     parsed_as: str) -> str:
    """只登记，不处理。返回 document_id，真正的向量化交给 process_document。

    上传接口必须立刻返回：向量化是分钟级的，同步阻塞在生产上会被反代掐断。
    """
    if source_id:
        with vector_db() as cur:
            _require_writable_source(cur, user, source_id)
        target = source_id
    else:
        target = str(ensure_default_source(user)["id"])

    content_hash = str(extract_document_metadata(filename, text)["content_hash"])
    with vector_db() as cur:
        # 没有数据库 unique 约束是为了兼容已有库中的历史重复记录；事务级
        # advisory lock 仍能让同一 source/filename 的并发上传串行化。
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"source={target}|filename={filename}",),
        )
        cur.execute(
            """SELECT d.*, s.name AS source_name, s.owner_id
               FROM documents d JOIN sources s ON s.id = d.source_id
               WHERE d.source_id = %s AND d.filename = %s
               ORDER BY d.version DESC, d.created_at DESC
               LIMIT 1 FOR UPDATE OF d""",
            (target, filename),
        )
        existing = cur.fetchone()
        if existing is not None:
            if str(existing["content_hash"]) == content_hash:
                return str(existing["id"])
            if existing["status"] == "processing":
                raise ModuleError(
                    "同名文档正在处理，暂时不能更新",
                    code="document_busy", status=409,
                )

            # 更新使用同一个 document_id，权限引用和外部链接不失效；旧 dense、
            # sparse、parent 节点必须先清掉，避免两个版本同时参与检索。
            from . import chain
            chain.delete_document_nodes(
                owner_id=str(existing["owner_id"]),
                document_id=str(existing["id"]),
            )
            cur.execute(
                """INSERT INTO document_revisions
                     (document_id, version, content, content_hash, parsed_as,
                      char_count, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (document_id, version) DO NOTHING""",
                (existing["id"], existing["version"], existing["content"],
                 existing["content_hash"], existing["parsed_as"],
                 existing["char_count"], existing["updated_at"]),
            )
            cur.execute(
                """UPDATE documents
                   SET content = %s, content_hash = %s, version = version + 1,
                       status = %s, error = %s, parsed_as = %s,
                       char_count = %s, chunk_count = 0, attempt_count = 0,
                       available_at = now(), lease_until = NULL,
                       processing_started_at = NULL, finished_at = NULL,
                       updated_at = now()
                   WHERE id = %s""",
                (text, content_hash, status, error, parsed_as, len(text), existing["id"]),
            )
            cur.execute(
                """UPDATE wiki_pages
                   SET status = 'stale', error = NULL, updated_at = now()
                   WHERE document_id = %s""",
                (existing["id"],),
            )
            cur.execute(
                """UPDATE wiki_topics t
                   SET status = 'stale', error = NULL, updated_at = now()
                   WHERE EXISTS (
                     SELECT 1 FROM wiki_topic_documents r
                     WHERE r.topic_id = t.id AND r.document_id = %s
                   )""",
                (existing["id"],),
            )
            cur.execute(
                """INSERT INTO document_revisions
                     (document_id, version, content, content_hash, parsed_as, char_count)
                   SELECT id, version, content, content_hash, parsed_as, char_count
                   FROM documents WHERE id = %s
                   ON CONFLICT (document_id, version) DO NOTHING""",
                (existing["id"],),
            )
            return str(existing["id"])

        cur.execute(
            """
            INSERT INTO documents (source_id, filename, content, status,
                                   char_count, error, parsed_as, content_hash, version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
            RETURNING id
            """,
            (target, filename, text, status, len(text), error, parsed_as, content_hash),
        )
        document_id = str(cur.fetchone()["id"])
        cur.execute(
            """INSERT INTO document_revisions
                 (document_id, version, content, content_hash, parsed_as, char_count)
               VALUES (%s, 1, %s, %s, %s, %s)""",
            (document_id, text, content_hash, parsed_as, len(text)),
        )
        return document_id


# ---------------------------------------------------------------- Wiki 派生视图

def build_wiki_page(user: UserContext, document_id: str) -> dict:
    from .wiki import build_page
    return build_page(user, document_id)


def list_wiki_pages(user: UserContext) -> list[dict]:
    from .wiki import list_pages
    return list_pages(user)


def get_wiki_page(user: UserContext, page_id: str) -> dict:
    from .wiki import get_page
    return get_page(user, page_id)


def list_wiki_topics(user: UserContext) -> list[dict]:
    from .wiki import list_topics
    return list_topics(user)


def get_wiki_topic(user: UserContext, topic_id: str) -> dict:
    from .wiki import get_topic
    return get_topic(user, topic_id)


def get_wiki_health(user: UserContext) -> dict:
    from .wiki import wiki_health
    return wiki_health(user)


def run_wiki_lint(user: UserContext) -> dict:
    from .wiki import run_lint
    return run_lint(user)


def list_wiki_events(user: UserContext, limit: int = 20) -> list[dict]:
    from .wiki import list_events
    return list_events(user, limit=limit)


def search_wiki_pages(user: UserContext, query: str, top_k: int = 3) -> list[dict]:
    from .wiki import search_pages
    return search_pages(user, query, top_k=top_k)


def ask_wiki(user: UserContext, question: str) -> dict:
    from .wiki import answer_question
    return answer_question(user, question)


# ---------------------------------------------------------------- 交给框架
#
# 入库和检索都委托给 LlamaIndex（chain.py）。这两个函数在这里只是**转接**，
# 保留是因为 gateway 用 getattr 按名字派发，模块必须暴露 process / search。


def _claim_specific(document_id: str) -> bool:
    """原子领取指定文档，供 CLI/评测的同步 process 调用。"""
    lease = get_config().ingest_lease_seconds
    with vector_db() as cur:
        cur.execute(
            """WITH target AS (
                 SELECT d.id FROM documents d JOIN sources s ON s.id = d.source_id
                 WHERE d.id = %s AND (
                      (d.status = 'uploaded' AND d.available_at <= now())
                      OR (d.status = 'processing'
                          AND (d.lease_until IS NULL OR d.lease_until < now()))
                 )
                 FOR UPDATE OF d, s
               )
               UPDATE documents d
               SET status = 'processing', attempt_count = attempt_count + 1,
                   processing_started_at = now(),
                   lease_until = now() + make_interval(secs => %s),
                   updated_at = now()
               FROM target t WHERE d.id = t.id
               RETURNING d.id""",
            (document_id, lease),
        )
        return cur.fetchone() is not None


def claim_next() -> str | None:
    """用 SKIP LOCKED 原子领取一个任务；过期 processing 任务会被恢复。"""
    lease = get_config().ingest_lease_seconds
    with vector_db() as cur:
        cur.execute(
            """WITH candidate AS (
                 SELECT d.id FROM documents d JOIN sources s ON s.id = d.source_id
                 WHERE (d.status = 'uploaded' AND d.available_at <= now())
                    OR (d.status = 'processing'
                        AND (d.lease_until IS NULL OR d.lease_until < now()))
                 ORDER BY d.available_at, d.created_at
                 FOR UPDATE OF d, s SKIP LOCKED
                 LIMIT 1
               )
               UPDATE documents d
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


def renew_lease(document_id: str) -> bool:
    """处理中的 worker 心跳；已完成/被回收的任务不会被错误续租。"""
    lease = get_config().ingest_lease_seconds
    with vector_db() as cur:
        cur.execute(
            """UPDATE documents
               SET lease_until = now() + make_interval(secs => %s), updated_at = now()
               WHERE id = %s AND status = 'processing' RETURNING id""",
            (lease, document_id),
        )
        return cur.fetchone() is not None


def _record_failure(document_id: str, exc: Exception, attempt_count: int) -> None:
    cfg = get_config()
    code = getattr(exc, "code", type(exc).__name__)
    detail = getattr(exc, "message", str(exc))
    retryable = not isinstance(exc, ModuleError)
    payload = json.dumps({
        "code": code, "message": detail, "retryable": retryable,
        "attempt": attempt_count,
    }, ensure_ascii=False)
    with vector_db() as cur:
        if retryable and attempt_count < cfg.ingest_max_attempts:
            delay = cfg.ingest_retry_base_seconds * (2 ** max(attempt_count - 1, 0))
            cur.execute(
                """UPDATE documents
                   SET status = 'uploaded', error = %s,
                       available_at = now() + make_interval(secs => %s),
                       lease_until = NULL, updated_at = now()
                   WHERE id = %s AND status = 'processing'""",
                (payload, delay, document_id),
            )
        else:
            terminal = "dead_letter" if retryable else "failed"
            cur.execute(
                """UPDATE documents
                   SET status = %s, error = %s, lease_until = NULL,
                       finished_at = now(), updated_at = now()
                   WHERE id = %s AND status = 'processing'""",
                (terminal, payload, document_id),
            )


def process(document_id: str, *, claimed: bool = False) -> None:
    """后台处理：把已登记的文档交给 LlamaIndex 切分 + 向量化。

    ★ 状态机仍然由本文件管：任何失败都必须落 failed，绝不伪装成 ready。
      框架负责"怎么切怎么存"，但"这篇处理到哪一步了、为什么失败"是产品状态，
      框架不关心，得自己维护。
    """
    from . import chain

    if not claimed and not _claim_specific(document_id):
        # 重复投递、延迟尚未到或任务已经完成：全部幂等忽略。
        return
    attempt_count = 1
    try:
        with vector_db() as cur:
            cur.execute(
                """SELECT d.filename, d.content, s.name AS source_name,
                          s.visibility, s.owner_id, d.attempt_count
                   FROM documents d JOIN sources s ON s.id = d.source_id
                   WHERE d.id = %s AND d.status = 'processing'""", (document_id,))
            row = cur.fetchone()
            if row is None:
                return
            attempt_count = int(row["attempt_count"])

        if not (row["content"] or "").strip():
            raise ModuleError("文档正文为空", code="empty_document")

        if row["attempt_count"] > 1:
            # 前一次进程可能在 dense 写完、状态提交前崩溃；按 document_id
            # 清半成品后再做下一次，保证恢复不会复制节点。
            chain.delete_document_nodes(
                owner_id=str(row["owner_id"]), document_id=document_id)

        count = chain.ingest_raw(
            filename=row["filename"], text=row["content"],
            source_name=row["source_name"], visibility=row["visibility"],
            owner_id=str(row["owner_id"]), document_id=document_id)

        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'ready', error = NULL, "
                "chunk_count = %s, lease_until = NULL, finished_at = now(), "
                "updated_at = now() WHERE id = %s AND status = 'processing'",
                (count, document_id))
    except Exception as exc:                                      # noqa: BLE001
        _record_failure(document_id, exc, attempt_count)


def search(user: UserContext, query: str, top_k: int = 5, mode: str = "hybrid",
           *, rerank_pool: int = 0, reranker: str = "ce",
           candidate_pool: int = 0,
           use_mmr: bool = False, mmr_lambda: float = 0.7,
           metadata_filtering: bool = True,
           within_filename: str | None = None,
           within_source: str | None = None,
           expand_parent: bool | None = None,
           explain: bool = False):
    """检索入口。gateway 按名字派发到这里，评测脚本也走这条。

    ★ `rerank_pool` 必须在这一层透传，不能只留在 `chain.search` 上。

      迁移到 LlamaIndex 时这个参数在这里丢了，于是 `scripts/ablate_rerank.py`
      调 `core.search(..., rerank_pool=pool)` 直接 TypeError ——
      **重排的消融实验因此跑不起来，而且没人发现**（那份 json 是迁移前的数据）。

      教训：**评测脚本是 core 的调用方之一，改 core 的签名等于改评测的 API。**
      签名收窄不会在类型检查里报错，只会在下次跑评测时才炸。
    """
    from . import chain
    from .cache import retrieval_cache, retrieval_cache_key

    # explain 模式要展示每个阶段的真实耗时和候选，不读缓存。
    cache_key = None
    if not explain:
        cache_key = retrieval_cache_key(
            user_id=user.user_id, is_admin=user.is_admin,
            knowledge_version=knowledge_version(user), query=query, top_k=top_k,
            options=(mode, rerank_pool, reranker, candidate_pool, use_mmr,
                     mmr_lambda, metadata_filtering, within_filename, within_source,
                     expand_parent),
        )
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            return cached

    result = chain.search(
        user, query, top_k=top_k, mode=mode,
        rerank_pool=rerank_pool, reranker=reranker,
        candidate_pool=candidate_pool,
        use_mmr=use_mmr, mmr_lambda=mmr_lambda,
        metadata_filtering=metadata_filtering,
        within_filename=within_filename, within_source=within_source,
        expand_parent=expand_parent, explain=explain,
    )
    if cache_key is not None:
        retrieval_cache.put(cache_key, result)
    return result
