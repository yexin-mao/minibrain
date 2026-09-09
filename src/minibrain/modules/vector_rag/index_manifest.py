"""主向量索引的 embedding 生成签名与 fail-fast 校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from ...config import get_config
from ...contracts import ModuleError
from ...db import vector_index_db

INDEX_NAME = "nodes"
TRANSFORM_VERSION = "mrl-head-l2-v1"


@dataclass(frozen=True)
class IndexSignature:
    embedding_endpoint: str
    embedding_model: str
    embedding_dimensions: int
    transform_version: str = TRANSFORM_VERSION


def _endpoint_identity(url: str) -> str:
    """保留路由身份，但移除用户名、密码、query 与 fragment。"""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.casefold()}://{host.casefold()}{port}{path}"


def current_signature() -> IndexSignature:
    cfg = get_config()
    return IndexSignature(
        embedding_endpoint=_endpoint_identity(cfg.embedding_base_url),
        embedding_model=cfg.embedding_model.strip(),
        embedding_dimensions=cfg.embedding_dimensions,
    )


def signature_differences(stored: dict, current: IndexSignature) -> dict[str, dict]:
    expected = asdict(current)
    return {
        key: {"stored": stored.get(key), "configured": value}
        for key, value in expected.items() if stored.get(key) != value
    }


def _node_count(cur) -> int:
    cur.execute("SELECT to_regclass('data_nodes') AS table_name")
    if cur.fetchone()["table_name"] is None:
        return 0
    cur.execute("SELECT count(*) AS count FROM data_nodes")
    return int(cur.fetchone()["count"])


def _insert(cur, signature: IndexSignature, provenance: str) -> dict:
    cur.execute(
        """INSERT INTO index_manifest
             (index_name, embedding_endpoint, embedding_model,
              embedding_dimensions, transform_version, provenance)
           VALUES (%s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (INDEX_NAME, signature.embedding_endpoint, signature.embedding_model,
         signature.embedding_dimensions, signature.transform_version, provenance),
    )
    return dict(cur.fetchone())


def ensure_index_compatible() -> dict:
    """首次空索引自动登记；旧非空索引或配置漂移时拒绝读写。"""
    signature = current_signature()
    with vector_index_db() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("minibrain:vector-index-manifest:nodes",),
        )
        cur.execute("SELECT * FROM index_manifest WHERE index_name = %s", (INDEX_NAME,))
        row = cur.fetchone()
        if row is None:
            count = _node_count(cur)
            if count:
                raise ModuleError(
                    f"向量索引已有 {count} 个节点但缺少 embedding manifest；"
                    "不能自动猜测历史模型。请确认配置后显式 adopt，或清空并重建索引。",
                    code="embedding_manifest_missing", status=503,
                )
            return _insert(cur, signature, "initialized_empty")

        result = dict(row)
        differences = signature_differences(result, signature)
        if differences:
            fields = ", ".join(sorted(differences))
            raise ModuleError(
                f"当前 embedding 配置与已有索引不兼容（{fields}）。"
                "必须使用原配置，或清空后全量重建；禁止新旧向量混写。",
                code="embedding_index_mismatch", status=503,
            )
        return result


def adopt_current_index() -> dict:
    """显式确认迁移前索引由当前配置生成；不修改任何向量。"""
    signature = current_signature()
    with vector_index_db() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("minibrain:vector-index-manifest:nodes",),
        )
        cur.execute("SELECT * FROM index_manifest WHERE index_name = %s", (INDEX_NAME,))
        row = cur.fetchone()
        if row is not None:
            if signature_differences(dict(row), signature):
                raise ModuleError(
                    "manifest 已存在且与当前配置不一致，adopt 不能覆盖历史签名",
                    code="embedding_index_mismatch", status=503,
                )
            return dict(row)
        return _insert(cur, signature, "operator_adopted")


def manifest_status() -> dict:
    with vector_index_db() as cur:
        cur.execute("SELECT * FROM index_manifest WHERE index_name = %s", (INDEX_NAME,))
        row = cur.fetchone()
        count = _node_count(cur)
    current = current_signature()
    stored = dict(row) if row else None
    return {
        "node_count": count,
        "configured": asdict(current),
        "stored": stored,
        "compatible": bool(stored) and not signature_differences(stored, current),
        "requires_adoption": stored is None and count > 0,
    }


def manifest_health() -> dict:
    """面向公开 health 的最小状态，不暴露 embedding endpoint。"""
    status = manifest_status()
    stored = status["stored"] or {}
    return {
        "compatible": status["compatible"],
        "requires_adoption": status["requires_adoption"],
        "node_count": status["node_count"],
        "embedding_model": stored.get("embedding_model"),
        "embedding_dimensions": stored.get("embedding_dimensions"),
        "provenance": stored.get("provenance"),
    }


def clear_manifest() -> None:
    with vector_index_db() as cur:
        cur.execute("DELETE FROM index_manifest WHERE index_name = %s", (INDEX_NAME,))
