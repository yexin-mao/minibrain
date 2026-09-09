from __future__ import annotations

from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

from minibrain.contracts import Evidence
from minibrain.modules.vector_rag import chain


def _node(node_id: str, text: str, score: float) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            id_=node_id, text=text,
            metadata={"source_name": "制度", "filename": "policy.md", "ordinal": 1},
        ),
        score=score,
    )


def test_hybrid_trace_keeps_dense_score_before_rrf_mutates_node(monkeypatch):
    dense = [_node("dense", "语义命中", 0.91)]
    keyword = [_node("keyword", "编号命中", 3.7)]

    class FakeRetriever:
        def __init__(self, **kwargs):
            pass

        def retrieve(self, query):
            return dense

    monkeypatch.setattr(chain, "VectorIndexRetriever", FakeRetriever)
    monkeypatch.setattr(chain, "get_index", lambda: object())
    monkeypatch.setattr(chain, "_keyword_nodes", lambda *args, **kwargs: keyword)

    stages = []
    result = chain._retrieve_nodes(
        SimpleNamespace(), "住宿标准", mode="hybrid", num_queries=1,
        fetch_k=10, filters=None, business={}, stages=stages,
    )

    assert [stage.name for stage in stages] == ["Dense 召回", "BM25 召回", "RRF 融合"]
    assert stages[0].candidates[0].score == 0.91
    assert stages[1].candidates[0].score == 3.7
    assert stages[2].score_kind == "RRF"
    assert result[0].score != 0.91  # RRF 会就地覆盖节点分数，但不能污染早期快照


def test_search_builds_trace_only_when_explain_is_enabled(monkeypatch):
    def fake_retrieve(*args, stages=None, **kwargs):
        nodes = [_node("n1", "每天不超过 500 元", 0.8)]
        if stages is not None:
            chain._add_stage(stages, "Dense 召回", "向量相似度", nodes, 0.0)
        return nodes

    monkeypatch.setattr(chain, "_retrieve_nodes", fake_retrieve)
    monkeypatch.setattr(chain, "extract_query_metadata", lambda query: {})
    monkeypatch.setattr(chain, "build_filters", lambda *args, **kwargs: None)
    monkeypatch.setattr(chain, "_evidence_from_ranked_nodes", lambda *args, **kwargs: [
        Evidence("vector-rag", "制度", "policy.md #1", "每天不超过 500 元", 0.8)
    ])
    monkeypatch.setattr(
        chain, "get_config",
        lambda: SimpleNamespace(expand_parent_context=False, rerank_ce_model="test"),
    )

    normal = chain.search(SimpleNamespace(), "住宿标准")
    explained = chain.search(SimpleNamespace(), "住宿标准", explain=True)

    assert normal.retrieval_trace is None
    assert normal.retrieval_signals is not None
    assert explained.retrieval_trace is not None
    assert explained.retrieval_signals is not None
    assert [stage.name for stage in explained.retrieval_trace.stages] == [
        "Dense 召回", "确定性去重", "Parent 展开与最终截断",
    ]
    assert explained.retrieval_trace.fetch_k == 10
    assert explained.retrieval_signals.dense_top_score == 0.8
    assert explained.retrieval_signals.calibration_status in {
        "report_only", "calibrated", "deterministic"}
