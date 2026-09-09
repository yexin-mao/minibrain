from minibrain.contracts import RetrievalCandidate, RetrievalStage, RetrievalTrace
from minibrain.modules.vector_rag.confidence import (
    classify_retrieval,
    extract_retrieval_signals,
)


def _candidate(rank: int, node_id: str, score: float):
    return RetrievalCandidate(
        rank=rank, node_id=node_id, source_name="s", location="f#1",
        snippet=node_id, score=score,
    )


def _stage(name: str, scores: list[tuple[str, float]]):
    return RetrievalStage(
        name=name, score_kind=name,
        candidates=[_candidate(i, node, score)
                    for i, (node, score) in enumerate(scores, start=1)],
        latency_ms=1,
    )


def test_extracts_raw_signals_and_applies_report_only_policy():
    trace = RetrievalTrace(
        query="q", retrieval_query="q", mode="hybrid", fetch_k=10,
        business_filters={}, reranker=None, mmr_lambda=None,
        stages=[
            _stage("Dense 召回", [("a", 0.9), ("b", 0.7), ("c", 0.6)]),
            _stage("BM25 召回", [("b", 4.2), ("d", 3.0)]),
            _stage("RRF 融合", [("b", 0.03), ("a", 0.02)]),
            _stage("Parent 展开与最终截断", [("e1", 0.03)]),
        ],
    )

    signals = extract_retrieval_signals(trace)

    assert signals.dense_top_score == 0.9
    assert signals.dense_margin == 0.2
    assert signals.keyword_top_score == 4.2
    assert signals.dense_keyword_overlap_at_5 == 0.5
    assert signals.fused_top_score == 0.03
    assert signals.final_evidence_count == 1
    assert signals.calibration_status == "report_only"
    assert signals.decision == "evidence_found"


def test_exact_identifier_metadata_miss_is_deterministic_no_evidence():
    trace = RetrievalTrace(
        query="PRJ-2026-0999 的负责人", retrieval_query="PRJ-2026-0999 的负责人",
        mode="hybrid", fetch_k=10,
        business_filters={"document_refs": ["PRJ-2026-0999"]},
        reranker=None, mmr_lambda=None,
        stages=[
            _stage("Dense 召回", []),
            _stage("Dense 召回（移除业务 metadata 后回退）", [("similar", 0.9)]),
            _stage("Parent 展开与最终截断", [("e1", 0.9)]),
        ],
    )
    signals = extract_retrieval_signals(trace)
    assert signals.decision == "no_evidence"
    assert signals.calibration_status == "deterministic"
    assert "精确业务编号" in signals.decision_reason


def test_low_score_policy_stays_uncertain_when_not_enabled():
    trace = RetrievalTrace(
        query="q", retrieval_query="q", mode="hybrid", fetch_k=10,
        business_filters={}, reranker=None, mmr_lambda=None,
        stages=[
            _stage("Dense 召回", [("a", 0.2)]),
            _stage("BM25 召回", []),
            _stage("Parent 展开与最终截断", [("e1", 0.2)]),
        ],
    )
    raw = extract_retrieval_signals(trace)
    signals = classify_retrieval(trace, raw, policy={
        "version": "test", "score_gate_enabled": False,
        "dense_no_evidence_below": 0.3,
    })
    assert signals.decision == "uncertain"
    assert signals.calibration_status == "report_only"
