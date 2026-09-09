"""结构感知切分与 small-to-big 展开。"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from minibrain.modules.vector_rag import chain
from minibrain.modules.vector_rag.hierarchy import build_hierarchy, markdown_sections


def test_markdown_sections_keep_heading_path_and_ignore_fenced_headings():
    sections = markdown_sections(
        "# 手册\n开头\n## 报销\n标准正文\n```md\n# 代码里的标题\n```\n## 请假\n请假正文")
    assert [path for path, _ in sections] == ["手册", "手册 > 报销", "手册 > 请假"]
    assert "# 代码里的标题" in sections[1][1]


def test_consecutive_headings_do_not_create_title_only_chunks():
    children, _ = build_hierarchy(
        filename="policy.md",
        text="# 供应商管理手册\n\n## 签约前检查\n\n签约前必须完成采购比价。",
        source_name="知识库", visibility="private", owner_id="u1",
        document_id="d1", child_size=100, child_overlap=0, parent_size=300,
        document_metadata={},
    )

    assert len(children) == 1
    assert "供应商管理手册 > 签约前检查" in children[0].get_content()
    assert "采购比价" in children[0].get_content()


def test_children_point_to_larger_persistable_parent():
    children, parents = build_hierarchy(
        filename="policy.md", text="# 差旅\n" + "住宿标准。" * 300,
        source_name="handbook", visibility="private", owner_id="u1",
        document_id="d1", child_size=100, child_overlap=0, parent_size=300,
        document_metadata={"document_kind": "policy"},
    )
    assert len(children) > len(parents) >= 1
    parent_ids = {parent.id for parent in parents}
    assert {node.metadata["parent_context_id"] for node in children} <= parent_ids
    assert {node.metadata["heading_path"] for node in children} == {"差旅"}
    assert all(node.metadata["node_level"] == "child" for node in children)
    assert all("标题路径：差旅" in node.get_content() for node in children)


def test_pdf_page_and_pptx_slide_headings_become_typed_metadata():
    pdf_nodes, _ = build_hierarchy(
        filename="report.pdf", text="# 第 12 页\n\n本页结论。",
        source_name="reports", visibility="private", owner_id="u1",
        document_id="d1", child_size=100, child_overlap=0, parent_size=300,
        document_metadata={"file_type": "pdf"},
    )
    slide_nodes, _ = build_hierarchy(
        filename="review.pptx", text="# 幻灯片 3\n\n季度结论。",
        source_name="slides", visibility="private", owner_id="u1",
        document_id="d2", child_size=100, child_overlap=0, parent_size=300,
        document_metadata={"file_type": "pptx"},
    )
    assert pdf_nodes[0].metadata["page_number"] == 12
    assert "slide_number" not in pdf_nodes[0].metadata
    assert slide_nodes[0].metadata["slide_number"] == 3


def test_parent_expansion_deduplicates_sibling_children(monkeypatch, alice):
    nodes = [
        NodeWithScore(node=TextNode(text="child one", id_="c1", metadata={
            "parent_context_id": "p1", "filename": "policy.md", "ordinal": 0,
            "source_name": "s"}), score=0.9),
        NodeWithScore(node=TextNode(text="child two", id_="c2", metadata={
            "parent_context_id": "p1", "filename": "policy.md", "ordinal": 1,
            "source_name": "s"}), score=0.8),
    ]
    monkeypatch.setattr(chain, "_load_visible_parents", lambda *_: {
        "p1": {"id": "p1", "filename": "policy.md", "ordinal": 0,
               "heading_path": "差旅 > 住宿", "content": "完整父块", "source_name": "s"},
    })
    evidence = chain._evidence_from_ranked_nodes(alice, nodes, 5)
    assert len(evidence) == 1
    assert evidence[0].snippet == "完整父块"
    assert "差旅 > 住宿" in evidence[0].location


def test_flat_arm_returns_same_ranked_children_without_parent_lookup(monkeypatch, alice):
    nodes = [NodeWithScore(node=TextNode(text=f"child {i}", id_=f"c{i}", metadata={
        "parent_context_id": "p1", "filename": "policy.md", "ordinal": i,
        "source_name": "s"}), score=1 - i / 10) for i in range(2)]
    monkeypatch.setattr(chain, "_load_visible_parents", lambda *_: (_ for _ in ()).throw(
        AssertionError("flat 组不应读取 parent 表")))
    evidence = chain._evidence_from_ranked_nodes(
        alice, nodes, 2, expand_parent=False)
    assert [item.snippet for item in evidence] == ["child 0", "child 1"]
    assert [item.location for item in evidence] == ["policy.md #0", "policy.md #1"]

    deduped = chain._evidence_from_ranked_nodes(
        alice, nodes, 2, expand_parent=False, deduplicate_parent=True)
    assert [item.snippet for item in deduped] == ["child 0"]
