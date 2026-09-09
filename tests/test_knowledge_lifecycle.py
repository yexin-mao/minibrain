"""知识库生命周期：权限、状态约束、幂等和物理清理。"""

from __future__ import annotations

import pytest

from minibrain import gateway
from minibrain.contracts import ModuleError, PermissionDenied
from minibrain.db import table_db, vector_db
from minibrain.modules.vector_rag import chain, core as vector_core


def _document(user, document_id: str) -> dict:
    return next(d for d in gateway.call("vector-rag", "list_documents", user)
                if str(d["id"]) == document_id)


def _dataset(user, dataset_id: str) -> dict:
    return next(d for d in gateway.call("table-rag", "list_datasets", user)
                if str(d["id"]) == dataset_id)


def test_owner_can_delete_uploaded_document(alice, monkeypatch):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "delete-me.md", "正文")
    deleted: list[str] = []
    monkeypatch.setattr(
        chain, "delete_document_nodes",
        lambda **kwargs: deleted.append(kwargs["document_id"]),
    )

    gateway.call("vector-rag", "delete_document", alice, document_id)

    assert deleted == [document_id]
    assert all(str(row["id"]) != document_id
               for row in gateway.call("vector-rag", "list_documents", alice))


def test_stranger_cannot_delete_document(alice, bob):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "private-delete.md", "正文")
    with pytest.raises(PermissionDenied):
        gateway.call("vector-rag", "delete_document", bob, document_id)


def test_processing_document_cannot_be_deleted(alice):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "busy.md", "正文")
    with vector_db() as cur:
        cur.execute("UPDATE documents SET status = 'processing' WHERE id = %s", (document_id,))

    try:
        with pytest.raises(ModuleError) as caught:
            gateway.call("vector-rag", "delete_document", alice, document_id)
        assert caught.value.code == "document_busy"
    finally:
        # 不把人为制造的 processing 状态泄漏给 session fixture 的 purge。
        with vector_db() as cur:
            cur.execute("UPDATE documents SET status = 'failed' WHERE id = %s", (document_id,))


def test_retry_failed_document_returns_to_uploaded(alice, monkeypatch):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "retry.md", "可重试正文")
    with vector_db() as cur:
        cur.execute(
            "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
            ('{"code":"timeout","message":"timeout"}', document_id),
        )
    monkeypatch.setattr(chain, "delete_document_nodes", lambda **kwargs: None)

    assert gateway.call("vector-rag", "retry_document", alice, document_id) == document_id
    row = _document(alice, document_id)
    assert row["status"] == "uploaded"
    assert row["error"] is None


def test_process_is_idempotent_after_ready(alice, monkeypatch):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "once.md", "只应入库一次")
    calls: list[str] = []

    def fake_ingest(**kwargs):
        calls.append(kwargs["document_id"])
        return 2

    monkeypatch.setattr(chain, "ingest_raw", fake_ingest)
    vector_core.process(document_id)
    vector_core.process(document_id)

    assert calls == [document_id]
    assert _document(alice, document_id)["chunk_count"] == 2


def test_same_document_upload_is_idempotent_and_changed_content_bumps_version(
        alice, monkeypatch):
    deleted: list[str] = []
    monkeypatch.setattr(
        chain, "delete_document_nodes",
        lambda **kwargs: deleted.append(kwargs["document_id"]),
    )

    first = gateway.call(
        "vector-rag", "upload_document", alice, None, "versioned.md", "第一版正文")
    duplicate = gateway.call(
        "vector-rag", "upload_document", alice, None, "versioned.md", "第一版正文")
    updated = gateway.call(
        "vector-rag", "upload_document", alice, None, "versioned.md", "第二版正文")

    assert duplicate == first
    assert updated == first
    row = _document(alice, first)
    assert row["version"] == 2
    assert len(row["content_hash"]) == 64
    assert row["status"] == "uploaded"
    assert deleted == [first]


def test_table_dataset_delete_drops_physical_table(alice):
    dataset_id = gateway.call(
        "table-rag", "upload_csv", alice, None, "delete.csv", b"name,value\na,1\n")
    gateway.process("table-rag", dataset_id)
    table_name = _dataset(alice, dataset_id)["table_name"]

    gateway.call("table-rag", "delete_dataset", alice, dataset_id)

    assert all(str(row["id"]) != dataset_id
               for row in gateway.call("table-rag", "list_datasets", alice))
    with table_db() as cur:
        cur.execute("SELECT to_regclass(%s) AS relation", (table_name,))
        assert cur.fetchone()["relation"] is None


def test_failed_dataset_can_be_retried(alice):
    dataset_id = gateway.call(
        "table-rag", "upload_csv", alice, None, "broken.csv", b"only_header\n")
    gateway.process("table-rag", dataset_id)
    assert _dataset(alice, dataset_id)["status"] == "failed"

    gateway.call("table-rag", "retry_dataset", alice, dataset_id)
    row = _dataset(alice, dataset_id)
    assert row["status"] == "uploaded"
    assert row["error"] is None


def test_table_source_creation_has_same_public_permission_rule(bob):
    with pytest.raises(PermissionDenied):
        gateway.call("table-rag", "create_source", bob, "not-public", "public")
