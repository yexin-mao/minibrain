"""回答支持安全门：纯确定性检查，不调用 LLM。"""

from minibrain.agent.citations import ClaimOutput, validate_claims
from minibrain.agent.confidence import (
    SAFE_REFUSAL,
    apply_confidence_gate,
    assess_answer_support,
)
from minibrain.contracts import Evidence


def _evidence(text: str = "差旅住宿标准为每天 500 元，制度编号 TR-2026-01。"):
    return [Evidence(
        module="vector-rag", source_name="差旅制度", location="policy.md#1",
        snippet=text, evidence_id="E1.1",
    )]


def _report(claims: list[ClaimOutput], evidence=None):
    evidence = _evidence() if evidence is None else evidence
    checked, metrics = validate_claims(claims, evidence)
    return assess_answer_support(checked, metrics, evidence)


def test_valid_citation_and_discrete_facts_pass():
    report = _report([ClaimOutput(
        text="住宿标准为每天 500 元（TR-2026-01）。",
        evidence_ids=["E1.1"],
    )])

    assert report.status == "deterministic_pass"
    assert report.blocked is False
    assert report.supported_claim_count == 1
    assert apply_confidence_gate("原答案", report) == "原答案"


def test_uncited_claim_is_blocked():
    report = _report([ClaimOutput(
        text="住宿标准为每天 500 元。", evidence_ids=[],
    )])

    assert report.blocked is True
    assert any("没有有效引用" in reason for reason in report.reasons)
    assert apply_confidence_gate("未经支持的答案", report) == SAFE_REFUSAL


def test_nonexistent_citation_is_blocked():
    report = _report([ClaimOutput(
        text="住宿标准为每天 500 元。", evidence_ids=["E9.9"],
    )])

    assert report.blocked is True
    assert any("不存在的证据" in reason for reason in report.reasons)


def test_fabricated_number_is_blocked_even_with_valid_citation():
    report = _report([ClaimOutput(
        text="住宿标准为每天 800 元。", evidence_ids=["E1.1"],
    )])

    assert report.blocked is True
    assert any("800" in reason for reason in report.reasons)


def test_qualitative_semantics_are_explicitly_not_claimed_as_verified():
    report = _report([ClaimOutput(
        text="这项制度比较宽松。", evidence_ids=["E1.1"],
    )])

    assert report.blocked is False
    assert report.scope == "citation_ids_and_discrete_facts_only"


def test_no_claim_and_no_evidence_is_treated_as_refusal_not_failure():
    report = _report([], evidence=[])

    assert report.status == "refusal_or_no_claims"
    assert report.blocked is False
