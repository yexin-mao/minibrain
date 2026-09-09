"""结构化 claim 与确定性引用校验。不调用模型，不碰数据库。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field

from ..contracts import Evidence


class ClaimOutput(BaseModel):
    """最终模型必须为每个事实性结论返回的结构。"""

    text: str = Field(description="一个原子事实性结论，不要把多个事实合并")
    evidence_ids: list[str] = Field(
        description="支持该结论的证据编号，例如 E1.1；没有证据时必须为空列表")


class StructuredAnswer(BaseModel):
    """Agent 的结构化最终输出；这是现有最终生成调用的输出格式。"""

    answer: str = Field(description="给用户看的简洁中文回答")
    claims: list[ClaimOutput] = Field(
        description="回答中的事实性结论；纯拒答或没有事实时可以为空")


def complete_truncated_answer(answer: str, claims: list[ClaimOutput]) -> str:
    """结构化正文意外停在冒号时，用已提交的原子 claim 确定性收口。

    某些 OpenAI-compatible 模型会成功提交完整 claims，却把给用户看的 answer
    截在“情况如下：”。claims 已经是同一次结构化输出的一部分，不需要再调用
    一个 LLM；同时保留引用编号，避免恢复出的正文脱离证据。
    """
    clean = answer.strip()
    if not claims or not clean.endswith(("：", ":")):
        return clean
    rendered = []
    for claim in claims:
        citations = "、".join(claim.evidence_ids)
        suffix = f"（证据 {citations}）" if citations else ""
        rendered.append(f"{claim.text}{suffix}")
    return f"{clean}\n" + "；\n".join(rendered) + "。"


@dataclass
class AnswerClaim:
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    valid_evidence_ids: list[str] = field(default_factory=list)
    invalid_evidence_ids: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    ungrounded_identifiers: list[str] = field(default_factory=list)
    grounded: bool | None = None


@dataclass
class CitationMetrics:
    claim_count: int = 0
    cited_claim_count: int = 0
    valid_reference_count: int = 0
    reference_count: int = 0
    citation_coverage: float | None = None
    citation_validity: float | None = None
    grounded_fact_rate: float | None = None


_GROUPED_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_IDENTIFIER = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+|[A-Z]{2,}")


def _normalise_number(raw: str) -> str:
    value = raw.replace(",", "")
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        return value
    return format(decimal.normalize(), "f")


def _numbers(text: str) -> list[str]:
    grouped = _GROUPED_NUMBER.findall(text)
    remaining = _GROUPED_NUMBER.sub(" ", text)
    values = [_normalise_number(item) for item in grouped + _NUMBER.findall(remaining)]
    return list(dict.fromkeys(values))


def _identifiers(text: str) -> list[str]:
    found: dict[str, str] = {}
    for item in _IDENTIFIER.findall(text):
        found.setdefault(item.upper(), item)
    return list(found.values())


def validate_claims(raw_claims: list[ClaimOutput], evidence: list[Evidence]) -> tuple[
        list[AnswerClaim], CitationMetrics]:
    """校验 ID 存在性，以及 claim 中数字/编号是否出现在它引用的原文中。"""
    evidence_by_id = {
        item.evidence_id: item for item in evidence if item.evidence_id is not None
    }
    claims: list[AnswerClaim] = []
    grounded_atoms = 0
    total_atoms = 0

    for raw in raw_claims:
        requested = list(dict.fromkeys(raw.evidence_ids))
        valid = [item for item in requested if item in evidence_by_id]
        invalid = [item for item in requested if item not in evidence_by_id]
        support = "\n".join(evidence_by_id[item].snippet for item in valid)
        support_numbers = set(_numbers(support))
        support_identifiers = {item.upper() for item in _identifiers(support)}
        numbers = _numbers(raw.text)
        identifiers = _identifiers(raw.text)
        missing_numbers = [item for item in numbers if item not in support_numbers]
        missing_identifiers = [
            item for item in identifiers if item.upper() not in support_identifiers]
        atom_count = len(numbers) + len(identifiers)
        missing_count = len(missing_numbers) + len(missing_identifiers)
        total_atoms += atom_count
        grounded_atoms += atom_count - missing_count
        claims.append(AnswerClaim(
            text=raw.text,
            evidence_ids=requested,
            valid_evidence_ids=valid,
            invalid_evidence_ids=invalid,
            ungrounded_numbers=missing_numbers,
            ungrounded_identifiers=missing_identifiers,
            grounded=(missing_count == 0) if atom_count else None,
        ))

    reference_count = sum(len(item.evidence_ids) for item in claims)
    valid_reference_count = sum(len(item.valid_evidence_ids) for item in claims)
    cited_claim_count = sum(bool(item.valid_evidence_ids) for item in claims)
    claim_count = len(claims)
    metrics = CitationMetrics(
        claim_count=claim_count,
        cited_claim_count=cited_claim_count,
        valid_reference_count=valid_reference_count,
        reference_count=reference_count,
        citation_coverage=(cited_claim_count / claim_count) if claim_count else None,
        citation_validity=(valid_reference_count / reference_count)
        if reference_count else None,
        grounded_fact_rate=(grounded_atoms / total_atoms) if total_atoms else None,
    )
    return claims, metrics
