"""Claim 级引用：结构化形状、ID 有效性和数字/编号落证据校验。"""

from __future__ import annotations

import json

from minibrain.agent.citations import (
    ClaimOutput,
    complete_truncated_answer,
    validate_claims,
)
from minibrain.agent.context import ContextAssembler
from minibrain.agent import graph
from minibrain.contracts import Evidence


def _evidence(evidence_id: str, snippet: str) -> Evidence:
    return Evidence(
        module="vector-rag", source_name="制度", location="policy.md#1",
        snippet=snippet, evidence_id=evidence_id,
    )


def test_valid_claim_has_full_citation_and_grounding():
    claims, metrics = validate_claims(
        [ClaimOutput(
            text="PRJ-2026-0142 的住宿标准是每天 500 元。",
            evidence_ids=["E1.1"],
        )],
        [_evidence("E1.1", "项目 PRJ-2026-0142：住宿标准 500.00 元/天。")],
    )

    assert claims[0].valid_evidence_ids == ["E1.1"]
    assert claims[0].grounded is True
    assert metrics.citation_coverage == 1.0
    assert metrics.citation_validity == 1.0
    assert metrics.grounded_fact_rate == 1.0


def test_fabricated_reference_and_number_are_reported_separately():
    claims, metrics = validate_claims(
        [ClaimOutput(text="住宿标准是每天 999 元。", evidence_ids=["E1.1", "E9.9"])],
        [_evidence("E1.1", "住宿标准是每天 500 元。")],
    )

    assert claims[0].invalid_evidence_ids == ["E9.9"]
    assert claims[0].ungrounded_numbers == ["999"]
    assert claims[0].grounded is False
    assert metrics.citation_coverage == 1.0
    assert metrics.citation_validity == 0.5
    assert metrics.grounded_fact_rate == 0.0


def test_uncited_claim_reduces_coverage_but_does_not_invent_validity():
    _, metrics = validate_claims(
        [ClaimOutput(text="有制度。", evidence_ids=["E1.1"]),
         ClaimOutput(text="另一个结论。", evidence_ids=[])],
        [_evidence("E1.1", "有制度。")],
    )

    assert metrics.citation_coverage == 0.5
    assert metrics.citation_validity == 1.0


def test_reject_answer_has_no_fake_perfect_score():
    claims, metrics = validate_claims([], [])
    assert claims == []
    assert metrics.citation_coverage is None
    assert metrics.citation_validity is None
    assert metrics.grounded_fact_rate is None


def test_identifier_requires_exact_token_not_substring():
    claims, _ = validate_claims(
        [ClaimOutput(text="SLA 已定义。", evidence_ids=["E1.1"])],
        [_evidence("E1.1", "这里只出现了 SLACK，不是目标缩写。")],
    )
    assert claims[0].ungrounded_identifiers == ["SLA"]


def test_complete_truncated_answer_uses_structured_claims_and_citations():
    answer = complete_truncated_answer(
        "根据现有信息，情况如下：",
        [
            ClaimOutput(text="制度上限为 400 元", evidence_ids=["E2.1"]),
            ClaimOutput(text="历史最高记录为 460 元", evidence_ids=["E1.1"]),
        ],
    )

    assert answer == (
        "根据现有信息，情况如下：\n"
        "制度上限为 400 元（证据 E2.1）；\n"
        "历史最高记录为 460 元（证据 E1.1）。"
    )


def test_complete_truncated_answer_leaves_complete_answer_unchanged():
    claims = [ClaimOutput(text="销售额为 3053000 元", evidence_ids=["E1.1"])]
    assert complete_truncated_answer("销售额为 3053000 元。", claims) == (
        "销售额为 3053000 元。")


def test_parallel_capable_tool_adapter_allocates_unique_call_prefixes(alice, monkeypatch):
    prefixes = []
    arguments_seen = []

    def fake_execute(user, name, arguments, *, evidence_prefix=None):
        prefixes.append(evidence_prefix)
        arguments_seen.append(json.loads(arguments))
        item = _evidence(f"{evidence_prefix}.1", f"证据 {evidence_prefix}")
        return f"[{item.evidence_id}] {item.snippet}", [item]

    monkeypatch.setattr(graph, "execute_tool", fake_execute)
    sink = []
    assembler = ContextAssembler(
        token_budget=1000, evidence_limit=10, candidate_pool=12)
    tool = graph._build_tools(alice, sink, assembler)[0]
    tool.invoke({"query": "第一问"})
    tool.invoke({"query": "第二问"})

    assert prefixes == ["E1", "E2"]
    assert [item["top_k"] for item in arguments_seen] == [12, 12]
    assert [item.evidence_id for item in sink] == ["E1.1", "E2.1"]
