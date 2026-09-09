from __future__ import annotations

from minibrain.contracts import UserContext
from minibrain.modules.vector_rag import chain, lexical
from minibrain.modules.vector_rag.metadata import (
    build_filters, extract_document_metadata, extract_query_metadata,
)


def test_document_metadata_extracts_stable_business_fields():
    metadata = extract_document_metadata(
        "meeting-mtg-20260617-02.md",
        "# Q2 会议\n关联 PRJ-2026-0204 和 TICKET-88407。",
    )
    assert metadata["document_kind"] == "meeting"
    assert metadata["file_type"] == "md"
    assert metadata["document_refs"] == [
        "MTG-20260617-02", "PRJ-2026-0204", "TICKET-88407",
    ]
    assert metadata["years"] == ["2026"]
    assert metadata["quarters"] == ["Q2"]
    assert len(metadata["content_hash"]) == 64


def test_content_hash_ignores_whitespace_only_changes():
    left = extract_document_metadata("a.md", "甲  乙\n丙")["content_hash"]
    right = extract_document_metadata("a.md", "甲 乙 丙")["content_hash"]
    assert left == right


def test_exact_reference_takes_precedence_over_looser_query_fields():
    extracted = extract_query_metadata("MTG-20260617-02 批准的项目负责人是谁？")
    assert extracted == {"document_refs": ["MTG-20260617-02"]}


def test_query_metadata_extracts_year_quarter_and_kinds():
    extracted = extract_query_metadata("2026 年 Q3 的项目报告")
    assert extracted == {
        "document_kind": ["project", "report"],
        "years": ["2026"],
        "quarters": ["Q3"],
    }


def test_filters_keep_permission_or_group_inside_business_and():
    user = UserContext(user_id="u1", username="alice", is_admin=False)
    filters = build_filters(user, "PRJ-2026-0142 是什么？")
    assert filters is not None
    assert filters.condition.value == "and"
    permission, business = filters.filters
    assert permission.condition.value == "or"
    assert business.condition.value == "or"
    assert business.filters[0].key == "document_refs"


def test_admin_without_business_filter_has_no_filter():
    admin = UserContext(user_id="admin", username="admin", is_admin=True)
    assert build_filters(admin, "普通问题") is None


def test_within_document_filter_keeps_permission_and_filename_in_sql_filter():
    user = UserContext(user_id="u1", username="alice", is_admin=False)
    filters = chain._within_document_filters(user, "policy-supplier.md")

    assert filters.condition.value == "and"
    permission, filename = filters.filters
    assert permission.condition.value == "or"
    assert filename.key == "filename"
    assert filename.value == "policy-supplier.md"

    where, params = lexical._where(user, {"filename": "policy-supplier.md"})
    assert "metadata_ ->>" in where
    assert params[-2:] == ["filename", ["policy-supplier.md"]]


def test_within_source_filter_keeps_permission_and_domain_in_sql_filter():
    user = UserContext(user_id="u1", username="alice", is_admin=False)
    filters = chain._within_source_filters(user, "研发制度")

    assert filters.condition.value == "and"
    permission, source = filters.filters
    assert permission.condition.value == "or"
    assert source.key == "source_name"
    assert source.value == "研发制度"
