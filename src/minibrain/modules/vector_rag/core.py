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

from ...contracts import ModuleError, NotFound, PermissionDenied, UserContext
from ...db import vector_db
from .loaders import UnsupportedFile, load_text

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
                   d.parsed_as, s.name AS source_name, s.visibility,
                   -- ★ 用缓存列而不是 count(chunks)：主路径的节点存在
                   -- LlamaIndex 自己的表里（mod_vector_li.data_nodes），
                   -- mod_vector.chunks 只有手写版才写。
                   d.chunk_count
            FROM documents d
            JOIN sources s ON s.id = d.source_id
            WHERE {where}
            ORDER BY d.created_at DESC
            LIMIT 100
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


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


def _first_heading(text: str) -> str | None:
    """取正文里第一个 markdown 标题作为文档主题。没有标题就返回 None。"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


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
    try:
        text, how = load_text(raw, filename)
    except UnsupportedFile as exc:
        error = str(exc)

    # error 列是 jsonb（结构化错误），和 process_document 里的失败写法保持一致：
    # {"code": ..., "message": ...}。前端和排查脚本只认这一种形状。
    payload = (json.dumps({"code": "unsupported_file", "message": error},
                          ensure_ascii=False) if error else None)
    doc_id = _insert_document(user, source_id, filename, text,
                              status="failed" if error else "uploaded",
                              error=payload, parsed_as=how)
    if error:
        raise ModuleError(error, code="unsupported_file", status=415)
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

    with vector_db() as cur:
        cur.execute(
            """
            INSERT INTO documents (source_id, filename, content, status,
                                   char_count, error, parsed_as)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (target, filename, text, status, len(text), error, parsed_as),
        )
        return str(cur.fetchone()["id"])


# ---------------------------------------------------------------- 交给框架
#
# 入库和检索都委托给 LlamaIndex（chain.py）。这两个函数在这里只是**转接**，
# 保留是因为 gateway 用 getattr 按名字派发，模块必须暴露 process / search。


def process(document_id: str) -> None:
    """后台处理：把已登记的文档交给 LlamaIndex 切分 + 向量化。

    ★ 状态机仍然由本文件管：任何失败都必须落 failed，绝不伪装成 ready。
      框架负责"怎么切怎么存"，但"这篇处理到哪一步了、为什么失败"是产品状态，
      框架不关心，得自己维护。
    """
    from . import chain

    try:
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'processing', updated_at = now() "
                "WHERE id = %s", (document_id,))
            cur.execute(
                """SELECT d.filename, d.content, s.name AS source_name,
                          s.visibility, s.owner_id
                   FROM documents d JOIN sources s ON s.id = d.source_id
                   WHERE d.id = %s""", (document_id,))
            row = cur.fetchone()
            if row is None:
                raise NotFound("document 不存在")

        if not (row["content"] or "").strip():
            raise ModuleError("文档正文为空", code="empty_document")

        count = chain.ingest_raw(
            filename=row["filename"], text=row["content"],
            source_name=row["source_name"], visibility=row["visibility"],
            owner_id=str(row["owner_id"]))

        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'ready', error = NULL, "
                "chunk_count = %s, updated_at = now() WHERE id = %s",
                (count, document_id))
    except Exception as exc:                                      # noqa: BLE001
        code = getattr(exc, "code", type(exc).__name__)
        detail = getattr(exc, "message", str(exc))
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', error = %s, "
                "updated_at = now() WHERE id = %s",
                (json.dumps({"code": code, "message": detail}, ensure_ascii=False),
                 document_id))


def search(user: UserContext, query: str, top_k: int = 5, mode: str = "hybrid"):
    from . import chain
    return chain.search(user, query, top_k=top_k, mode=mode)
