"""检索信号提取与三态 no-answer 策略。

这里故意不输出 high/low confidence：不同 embedding、语料和查询分布下，原始分数
不可直接比较。先记录信号，再用独立标注集校准阈值，才能决定是否启用拒答规则。
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from ...contracts import RetrievalSignals, RetrievalStage, RetrievalTrace

POLICY_PATH = Path(__file__).with_name("retrieval_confidence_policy.json")


@lru_cache(maxsize=1)
def load_policy() -> dict:
    """Load the reviewed calibration artifact bundled with the retrieval code."""
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": "missing", "score_gate_enabled": False}
    return policy if isinstance(policy, dict) else {
        "version": "invalid", "score_gate_enabled": False}


def _stage(trace: RetrievalTrace, prefix: str) -> RetrievalStage | None:
    # metadata 零命中回退会留下同名前缀的两组阶段；优先取最后一个有候选的阶段。
    matches = [stage for stage in trace.stages if stage.name.startswith(prefix)]
    return next((stage for stage in reversed(matches) if stage.candidates),
                matches[-1] if matches else None)


def _top_score(stage: RetrievalStage | None) -> float | None:
    if stage is None or not stage.candidates:
        return None
    return stage.candidates[0].score


def extract_retrieval_signals(trace: RetrievalTrace) -> RetrievalSignals:
    dense = _stage(trace, "Dense 召回")
    keyword = _stage(trace, "BM25 召回")
    fused = _stage(trace, "RRF 融合")
    final = _stage(trace, "Parent 展开与最终截断")

    dense_top = _top_score(dense)
    dense_margin = None
    if dense is not None and len(dense.candidates) >= 2:
        first, second = dense.candidates[:2]
        if first.score is not None and second.score is not None:
            dense_margin = round(first.score - second.score, 6)

    overlap = None
    if dense is not None and keyword is not None:
        dense_ids = {item.node_id for item in dense.candidates[:5]}
        keyword_ids = {item.node_id for item in keyword.candidates[:5]}
        denominator = min(5, len(dense_ids), len(keyword_ids))
        overlap = (
            round(len(dense_ids & keyword_ids) / denominator, 4)
            if denominator else 0.0
        )

    raw = RetrievalSignals(
        dense_top_score=dense_top,
        dense_margin=dense_margin,
        keyword_top_score=_top_score(keyword),
        dense_keyword_overlap_at_5=overlap,
        fused_top_score=_top_score(fused),
        final_evidence_count=len(final.candidates) if final else 0,
    )
    return classify_retrieval(trace, raw)


def _exact_reference_missed(trace: RetrievalTrace) -> bool:
    """An exact identifier filter missed and retrieval had to broaden the scope."""
    if not trace.business_filters.get("document_refs"):
        return False
    return any("移除业务 metadata 后回退" in stage.name for stage in trace.stages)


def classify_retrieval(
    trace: RetrievalTrace,
    signals: RetrievalSignals,
    *,
    policy: dict | None = None,
) -> RetrievalSignals:
    """Classify retrieval, never pretending relevance equals answerability.

    ``no_evidence`` is safe to enforce; ``uncertain`` must continue to the answer
    model and the claim-level grounding gate because a relevant document can still
    omit the requested attribute.
    """
    policy = load_policy() if policy is None else policy
    version = str(policy.get("version", "unknown"))
    if signals.final_evidence_count == 0:
        return replace(
            signals,
            policy_version=version,
            calibration_status="deterministic",
            decision="no_evidence",
            decision_reason="权限范围内没有返回任何证据",
        )
    if _exact_reference_missed(trace):
        return replace(
            signals,
            policy_version=version,
            calibration_status="deterministic",
            decision="no_evidence",
            decision_reason="查询中的精确业务编号在 metadata 索引中不存在",
        )

    threshold = policy.get("dense_no_evidence_below")
    if isinstance(threshold, (int, float)) and signals.dense_top_score is not None:
        below = signals.dense_top_score < float(threshold)
        no_keyword = signals.keyword_top_score is None or signals.keyword_top_score <= 0
        if below and no_keyword:
            enabled = bool(policy.get("score_gate_enabled", False))
            return replace(
                signals,
                policy_version=version,
                calibration_status="calibrated" if enabled else "report_only",
                decision="no_evidence" if enabled else "uncertain",
                decision_reason=(
                    f"Dense top1={signals.dense_top_score:.4f} 低于校准阈值 "
                    f"{float(threshold):.4f}，且 BM25 无命中"
                ),
            )

    return replace(
        signals,
        policy_version=version,
        calibration_status=(
            "calibrated" if policy.get("score_gate_enabled") else "report_only"),
        decision="evidence_found",
        decision_reason="检索信号未触发 no-evidence 规则；答案仍需引用安全门校验",
    )
