"""主路径全局 sparse 索引：召回独立性、权限、metadata 与切分确定性。"""

from __future__ import annotations

import json
import uuid

from llama_index.core.schema import NodeWithScore, TextNode

from minibrain.modules.vector_rag import chain, lexical
from minibrain.scripts_reindex_lexical import _attach_document_id
from minibrain.config import get_config
from minibrain.db import vector_index_db


def _node(text: str, *, owner_id: str, source_name: str,
          visibility: str = "private", **metadata) -> TextNode:
    return TextNode(text=text, metadata={
        "filename": f"{uuid.uuid4().hex}.md",
        "source_name": source_name,
        "owner_id": owner_id,
        "visibility": visibility,
        **metadata,
    })


def test_chunk_boundaries_do_not_depend_on_permission_metadata():
    text = "第一段。" * 400
    short = chain._prepare_nodes(
        [("a.md", text)], source_name="s", visibility="private", owner_id="1")
    long = chain._prepare_nodes(
        [("very-long-filename-" * 20 + ".md", text)],
        source_name="source/" + "x" * 200, visibility="private",
        owner_id="00000000-0000-0000-0000-000000000000")
    assert [node.get_content() for node in short] == [node.get_content() for node in long]


def test_registered_document_id_is_attached_after_splitting():
    nodes = chain._prepare_nodes(
        [("doc.md", "正文内容")], source_name="s", visibility="private",
        owner_id="owner", document_ids=["document-123"])
    assert nodes
    assert {node.metadata["registry_document_id"] for node in nodes} == {"document-123"}


def test_legacy_node_json_backfill_updates_both_metadata_copies():
    metadata = {
        "filename": "doc.md",
        "_node_content": '{"metadata":{"filename":"doc.md"},'
                         '"excluded_embed_metadata_keys":[]}',
    }
    updated = _attach_document_id(metadata, "document-123")
    nested = json.loads(updated["_node_content"])
    assert updated["registry_document_id"] == "document-123"
    assert nested["metadata"]["registry_document_id"] == "document-123"
    assert "registry_document_id" in nested["excluded_embed_metadata_keys"]


def test_document_delete_uses_non_reserved_metadata_key():
    """真实 PGVectorStore 回归：`document_id` 会被框架占用，业务键不能重名。"""
    owner_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    nodes, parents = chain._prepare_hierarchy(
        [("delete-exact.md", "精确删除测试")], source_name=f"delete/{uuid.uuid4()}",
        visibility="private", owner_id=owner_id, document_ids=[document_id])
    for node in nodes:
        node.embedding = [0.01] * get_config().embedding_dimensions
    node_ids = [node.node_id for node in nodes]

    try:
        chain._store_parent_contexts(parents)
        chain.get_index().vector_store.add(nodes)
        lexical.index_nodes(nodes)
        with vector_index_db() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM data_nodes "
                "WHERE metadata_->>'registry_document_id' = %s", (document_id,))
            assert cur.fetchone()["n"] == len(nodes)
            cur.execute("SELECT count(*) AS n FROM parent_contexts WHERE document_id = %s",
                        (document_id,))
            assert cur.fetchone()["n"] == len(parents)

        chain.delete_document_nodes(owner_id=owner_id, document_id=document_id)
        with vector_index_db() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM data_nodes "
                "WHERE metadata_->>'registry_document_id' = %s", (document_id,))
            assert cur.fetchone()["n"] == 0
            cur.execute("SELECT count(*) AS n FROM parent_contexts WHERE document_id = %s",
                        (document_id,))
            assert cur.fetchone()["n"] == 0
            cur.execute(
                "SELECT count(*) AS n FROM node_lexical_stats WHERE document_id = %s",
                (document_id,))
            assert cur.fetchone()["n"] == 0
    finally:
        # 断言中途失败也不能污染共享数据库。
        chain.get_index().vector_store.delete_nodes(node_ids=node_ids)
        with vector_index_db() as cur:
            cur.execute("DELETE FROM node_lexical_stats WHERE node_id = ANY(%s)", (node_ids,))
            cur.execute("DELETE FROM parent_contexts WHERE document_id = %s", (document_id,))


def test_rrf_can_add_document_missing_from_vector_candidates():
    vector = [NodeWithScore(node=TextNode(text="vector", id_="v"), score=0.9)]
    keyword = [NodeWithScore(node=TextNode(text="exact TICKET-90001", id_="k"), score=8.0)]
    fused = lexical.fuse_rankings(vector, keyword, 2)
    assert {item.node.node_id for item in fused} == {"v", "k"}


def test_keyword_mode_does_not_construct_vector_retriever(monkeypatch, alice):
    expected = [NodeWithScore(node=TextNode(text="exact", id_="k"), score=2.0)]
    monkeypatch.setattr(chain, "_keyword_nodes", lambda *args, **kwargs: expected)

    class MustNotConstruct:
        def __init__(self, *args, **kwargs):
            raise AssertionError("keyword 模式不应构造或调用向量检索器")

    monkeypatch.setattr(chain, "VectorIndexRetriever", MustNotConstruct)
    result = chain._retrieve_nodes(
        alice, "exact", mode="keyword", num_queries=1,
        fetch_k=5, filters=None, business={})
    assert result == expected


def test_global_sparse_index_respects_permissions_and_business_metadata(alice, bob):
    source = f"lexical-test/{uuid.uuid4().hex}"
    nodes = [
        _node("exacttoken alpha", owner_id=alice.user_id, source_name=source,
              document_kind="policy"),
        _node("exacttoken exacttoken beta", owner_id=alice.user_id, source_name=source,
              document_kind="meeting"),
    ]
    try:
        lexical.index_nodes(nodes)
        alice_hits = lexical.rank_node_ids(alice, "exacttoken", 10)
        assert {node_id for node_id, _ in alice_hits} == {node.node_id for node in nodes}
        assert lexical.rank_node_ids(bob, "exacttoken", 10) == []

        policy_hits = lexical.rank_node_ids(
            alice, "exacttoken", 10, business={"document_kind": ["policy"]})
        assert [node_id for node_id, _ in policy_hits] == [nodes[0].node_id]
    finally:
        lexical.delete_source(owner_id=alice.user_id, source_name=source)
