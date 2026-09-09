from __future__ import annotations

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from minibrain.modules.vector_rag.selection import deduplicate_nodes, mmr_select


def _node(text: str, filename: str, score: float, embedding: list[float], node_id: str):
    return NodeWithScore(
        node=TextNode(id_=node_id, text=text, metadata={"filename": filename},
                      embedding=embedding),
        score=score,
    )


def test_deduplicate_nodes_keeps_first_normalized_content():
    nodes = [
        _node("住宿上限 500 元", "a.md", 1.0, [1.0, 0.0], "1"),
        _node("住宿上限   500 元", "copy.md", 0.9, [1.0, 0.0], "2"),
        _node("交通费实报实销", "b.md", 0.8, [0.0, 1.0], "3"),
    ]
    assert [n.node.node_id for n in deduplicate_nodes(nodes)] == ["1", "3"]


def test_mmr_prefers_new_information_over_same_document_duplicate():
    nodes = [
        _node("技术部人数", "tech.md", 1.0, [1.0, 0.0], "1"),
        _node("技术部后端人数", "tech.md", 0.99, [0.99, 0.01], "2"),
        _node("销售部人数", "sales.md", 0.90, [0.0, 1.0], "3"),
    ]
    selected = mmr_select(nodes, 2, lambda_mult=0.7)
    assert [n.node.node_id for n in selected] == ["1", "3"]


def test_mmr_is_deterministic_on_ties():
    nodes = [
        _node("甲", "a.md", 1.0, [1.0, 0.0], "1"),
        _node("乙", "b.md", 1.0, [0.0, 1.0], "2"),
    ]
    assert [n.node.node_id for n in mmr_select(nodes, 1)] == ["1"]


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_mmr_rejects_invalid_lambda(bad):
    with pytest.raises(ValueError):
        mmr_select([], 5, lambda_mult=bad)
