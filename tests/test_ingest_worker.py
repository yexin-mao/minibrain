"""持久化入库队列：原子领取、lease 恢复、有限重试和 worker 分发。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from minibrain import gateway
from minibrain.db import table_db, vector_db
from minibrain.modules.table_rag import core as table_core
from minibrain.modules.vector_rag import chain, core as vector_core
from minibrain import scripts_worker


def _vector_row(document_id: str) -> dict:
    with vector_db() as cur:
        cur.execute(
            "SELECT status, attempt_count, available_at, lease_until, error "
            "FROM documents WHERE id = %s", (document_id,))
        return dict(cur.fetchone())


def _table_row(dataset_id: str) -> dict:
    with table_db() as cur:
        cur.execute(
            "SELECT status, attempt_count, available_at, lease_until, error "
            "FROM datasets WHERE id = %s", (dataset_id,))
        return dict(cur.fetchone())


def test_claim_next_executes_real_skip_locked_sql(alice):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "next.md", "正文")
    dataset_id = gateway.call(
        "table-rag", "upload_csv", alice, None, "next.csv", b"a,b\n1,2\n")
    try:
        assert vector_core.claim_next() == document_id
        assert table_core.claim_next() == dataset_id
        assert _vector_row(document_id)["status"] == "processing"
        assert _table_row(dataset_id)["status"] == "processing"
    finally:
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', lease_until = NULL WHERE id = %s",
                (document_id,))
        with table_db() as cur:
            cur.execute(
                "UPDATE datasets SET status = 'failed', lease_until = NULL WHERE id = %s",
                (dataset_id,))


def test_specific_claim_is_atomic_and_sets_lease(alice):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "claim.md", "正文")
    try:
        assert vector_core._claim_specific(document_id) is True
        assert vector_core._claim_specific(document_id) is False
        row = _vector_row(document_id)
        assert row["status"] == "processing"
        assert row["attempt_count"] == 1
        assert row["lease_until"] is not None
    finally:
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', lease_until = NULL WHERE id = %s",
                (document_id,))


def test_two_workers_cannot_claim_the_same_document(alice):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "two-workers.md", "正文")
    barrier = threading.Barrier(2)

    def claim() -> bool:
        barrier.wait()
        return vector_core._claim_specific(document_id)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _index: claim(), range(2)))
        assert sorted(results) == [False, True]
        assert _vector_row(document_id)["attempt_count"] == 1
    finally:
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', lease_until = NULL WHERE id = %s",
                (document_id,))


def test_expired_lease_can_be_reclaimed(alice):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "expired.md", "正文")
    assert vector_core._claim_specific(document_id)
    with vector_db() as cur:
        cur.execute(
            "UPDATE documents SET lease_until = now() - interval '1 second' WHERE id = %s",
            (document_id,))
    try:
        assert vector_core._claim_specific(document_id) is True
        assert _vector_row(document_id)["attempt_count"] == 2
    finally:
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', lease_until = NULL WHERE id = %s",
                (document_id,))


def test_transient_vector_failure_retries_then_stops(alice, monkeypatch):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "flaky.md", "正文")
    monkeypatch.setattr(chain, "ingest_raw", lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("temporary outage")))
    monkeypatch.setattr(chain, "delete_document_nodes", lambda **kwargs: None)

    for attempt in range(1, 4):
        with vector_db() as cur:
            cur.execute("UPDATE documents SET available_at = now() WHERE id = %s", (document_id,))
        vector_core.process(document_id)
        row = _vector_row(document_id)
        assert row["attempt_count"] == attempt
        if attempt < 3:
            assert row["status"] == "uploaded"
            assert row["error"]["retryable"] is True
        else:
            assert row["status"] == "dead_letter"
            assert row["lease_until"] is None


def test_heartbeat_renews_only_processing_tasks(alice):
    document_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "heartbeat.md", "正文")
    assert vector_core._claim_specific(document_id)
    assert vector_core.renew_lease(document_id) is True
    with vector_db() as cur:
        cur.execute("UPDATE documents SET status = 'failed' WHERE id = %s", (document_id,))
    assert vector_core.renew_lease(document_id) is False


def test_table_transient_failure_is_requeued(alice, monkeypatch):
    dataset_id = gateway.call(
        "table-rag", "upload_csv", alice, None, "flaky.csv", b"a,b\n1,2\n")
    monkeypatch.setattr(
        table_core.csv, "reader",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("temporary outage")),
    )

    table_core.process_dataset(dataset_id)
    row = _table_row(dataset_id)
    assert row["status"] == "uploaded"
    assert row["attempt_count"] == 1
    assert row["error"]["retryable"] is True
    assert row["lease_until"] is None


def test_worker_processes_at_most_one_job_per_module(monkeypatch):
    jobs = {"vector-rag": "doc-1", "table-rag": "table-1"}
    calls: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(scripts_worker.gateway, "claim_next", lambda module: jobs[module])
    monkeypatch.setattr(
        scripts_worker.gateway, "process",
        lambda module, entity, *, claimed=False: calls.append((module, entity, claimed)),
    )

    assert scripts_worker.run_once() == 2
    assert calls == [
        ("vector-rag", "doc-1", True),
        ("table-rag", "table-1", True),
    ]
