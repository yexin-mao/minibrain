"""证据上下文组装器：token 硬预算、跨调用去重和文档覆盖顺序。"""

from __future__ import annotations

from minibrain.agent.context import ContextAssembler
from minibrain.agent.tools import format_evidence
from minibrain.contracts import Evidence


def _evidence(evidence_id: str, location: str, snippet: str) -> Evidence:
    return Evidence(
        module="vector-rag", source_name="知识库", location=location,
        snippet=snippet, evidence_id=evidence_id,
    )


def _assembler(*, budget: int = 10000, limit: int = 12) -> ContextAssembler:
    return ContextAssembler(
        token_budget=budget, evidence_limit=limit, candidate_pool=12)


def test_first_result_stays_first_then_new_documents_are_covered():
    assembler = _assembler()
    selected = assembler.add([
        _evidence("E1.1", "a.md #1", "A 的第一段"),
        _evidence("E1.2", "a.md #2", "A 的第二段"),
        _evidence("E1.3", "b.md #1", "B 的第一段"),
    ])

    assert [item.evidence_id for item in selected] == ["E1.1", "E1.3", "E1.2"]
    assert [item.candidate_rank for item in assembler.decisions] == [1, 3, 2]
    assert assembler.metrics().context_tokens == assembler.count_tokens(
        format_evidence(selected))


def test_duplicate_is_removed_across_tool_calls():
    assembler = _assembler()
    assert assembler.add([_evidence("E1.1", "a.md #1", "同一段正文")])
    assert assembler.add([_evidence("E2.1", "a.md #1", "  同一段正文  ")]) == []

    metrics = assembler.metrics()
    assert metrics.candidate_count == 2
    assert metrics.selected_count == 1
    assert metrics.dropped_duplicate_count == 1


def test_token_budget_is_a_hard_limit():
    first = _evidence("E1.1", "a.md #1", "第一段内容")
    second = _evidence("E1.2", "b.md #1", "第二段完全不同的内容")
    probe = _assembler()
    first_tokens = probe._tokens_for(first)
    assembler = _assembler(budget=first_tokens)

    selected = assembler.add([first, second])

    assert [item.evidence_id for item in selected] == ["E1.1"]
    assert assembler.decisions[1].reason == "token_budget"
    assert assembler.metrics().context_tokens <= first_tokens
    assert assembler.metrics().budget_utilization == 1.0


def test_evidence_limit_is_independent_from_token_budget():
    assembler = _assembler(limit=1)
    selected = assembler.add([
        _evidence("E1.1", "a.md #1", "第一段"),
        _evidence("E1.2", "b.md #1", "第二段"),
    ])
    assert len(selected) == 1
    assert assembler.decisions[1].reason == "evidence_limit"
    assert assembler.metrics().dropped_limit_count == 1


def test_per_call_limit_reserves_capacity_for_a_followup_hop():
    assembler = _assembler(limit=6)
    first_hop = assembler.add([
        _evidence(f"E1.{i}", f"project-{i}.md #1", f"项目证据 {i}")
        for i in range(1, 7)
    ], max_items=4)
    second_hop = assembler.add([
        _evidence("E2.1", "dept-tech.md #1", "算法组属于技术部")
    ], max_items=4)

    assert len(first_hop) == 4
    assert [item.evidence_id for item in second_hop] == ["E2.1"]
    assert assembler.metrics().dropped_per_call_limit_count == 2
