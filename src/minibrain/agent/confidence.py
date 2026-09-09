"""零额外 LLM 的回答支持安全门。

检索相似度只能说明“像不像”，不能证明“能不能回答”。因此本模块不把未经
校准的余弦/RRF 分数伪装成 answerability probability，而是在生成后只拦截
可确定证明有问题的情况：无有效引用、伪造引用、数字或业务编号不在引用证据中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..contracts import Evidence
from .citations import AnswerClaim, CitationMetrics

ConfidenceStatus = Literal[
    "deterministic_pass", "blocked", "refusal_or_no_claims", "unverified"
]

SAFE_REFUSAL = (
    "现有证据不足以可靠支持该回答，我暂时无法确认。"
    "请补充相关资料，或把问题限定到知识库中已有的对象和时间范围。"
)


@dataclass(frozen=True)
class ConfidenceReport:
    status: ConfidenceStatus
    blocked: bool
    reasons: list[str] = field(default_factory=list)
    claim_count: int = 0
    supported_claim_count: int = 0
    evidence_count: int = 0
    # 明确写出口径，防止 UI/面试把确定性下限误称为语义 Faithfulness。
    scope: str = "citation_ids_and_discrete_facts_only"


def assess_answer_support(claims: list[AnswerClaim], metrics: CitationMetrics,
                          evidence: list[Evidence]) -> ConfidenceReport:
    """只依据确定性信号作结论；无法证明的语义支持明确标成 unverified。"""
    if not claims:
        status: ConfidenceStatus = (
            "refusal_or_no_claims" if not evidence else "unverified")
        reason = (
            "没有事实 claim 且没有证据，按纯拒答或无事实回答处理"
            if not evidence else
            "模型没有提交事实 claim，无法执行 claim 级确定性检查"
        )
        return ConfidenceReport(
            status=status, blocked=False, reasons=[reason],
            claim_count=0, supported_claim_count=0, evidence_count=len(evidence),
        )

    reasons: list[str] = []
    supported = 0
    for index, claim in enumerate(claims, start=1):
        prefix = f"claim {index}"
        claim_bad = False
        if not claim.valid_evidence_ids:
            reasons.append(f"{prefix} 没有有效引用")
            claim_bad = True
        if claim.invalid_evidence_ids:
            reasons.append(
                f"{prefix} 引用了不存在的证据：{', '.join(claim.invalid_evidence_ids)}")
            claim_bad = True
        if claim.ungrounded_numbers:
            reasons.append(
                f"{prefix} 的数字未落在引用证据中：{', '.join(claim.ungrounded_numbers)}")
            claim_bad = True
        if claim.ungrounded_identifiers:
            reasons.append(
                f"{prefix} 的编号未落在引用证据中：{', '.join(claim.ungrounded_identifiers)}")
            claim_bad = True
        if not claim_bad:
            supported += 1

    if reasons:
        return ConfidenceReport(
            status="blocked", blocked=True, reasons=reasons,
            claim_count=len(claims), supported_claim_count=supported,
            evidence_count=len(evidence),
        )

    # 引用存在、离散事实落证据不等于完整语义蕴含；名字刻意不用 faithful。
    return ConfidenceReport(
        status="deterministic_pass", blocked=False,
        reasons=["全部 claim 都有有效引用，数字和业务编号均落在对应证据中"],
        claim_count=len(claims), supported_claim_count=supported,
        evidence_count=len(evidence),
    )


def apply_confidence_gate(answer: str, report: ConfidenceReport) -> str:
    return SAFE_REFUSAL if report.blocked else answer
