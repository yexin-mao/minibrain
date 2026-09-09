from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minibrain.evaluation.answer_quality import (
    AnswerQualityJudgment,
    JsonlJudgeCache,
    LLMAnswerJudge,
    build_judge_payload,
    calibration_errors,
    quality_scores,
)


def _judgment(**updates):
    raw = {
        "answer_correctness": 0.75,
        "correctness_reason": "核心答案正确，但多说了一项",
        "answer_relevance": 1.0,
        "relevance_reason": "直接作答",
        "is_refusal": False,
        "refusal_reason": "给出了结论",
        "factual_claims": [
            {"claim": "住宿上限是 400 元", "verdict": "supported",
             "evidence_ids": ["E1.1"], "reason": "原文明确记载"},
            {"claim": "包含早餐", "verdict": "unsupported",
             "evidence_ids": [], "reason": "证据未提及"},
        ],
        "citation_judgments": [
            {"claim_index": 0, "verdict": "partially_supported", "reason": "金额支持，早餐不支持"},
        ],
    }
    raw.update(updates)
    return AnswerQualityJudgment.model_validate(raw)


def test_quality_scores_keep_dimensions_separate():
    scores = quality_scores(
        _judgment(), expected_refusal=False, submitted_claim_count=1)

    assert scores == {
        "answer_correctness": 0.75,
        "faithfulness": 0.5,
        "citation_correctness": 0.5,
        "answer_relevance": 1.0,
        "refusal_correct": True,
    }


def test_pure_refusal_has_no_fake_perfect_faithfulness():
    judgment = _judgment(
        is_refusal=True, factual_claims=[], citation_judgments=[])
    scores = quality_scores(
        judgment, expected_refusal=True, submitted_claim_count=0)

    assert scores["faithfulness"] is None
    assert scores["citation_correctness"] is None
    assert scores["refusal_correct"] is True


def test_not_applicable_refusal_claim_is_excluded_from_citation_score():
    judgment = _judgment(
        factual_claims=[{
            "claim": "当前资料未提供离职率", "verdict": "not_applicable",
            "evidence_ids": [], "reason": "信息缺失元声明",
        }],
        citation_judgments=[{
            "claim_index": 0, "verdict": "not_applicable",
            "reason": "正确拒答没有可引用正文",
        }],
    )
    scores = quality_scores(
        judgment, expected_refusal=True, submitted_claim_count=1)

    assert scores["citation_correctness"] is None
    assert scores["faithfulness"] is None


def test_calibration_reports_interpretable_mismatches():
    errors = calibration_errors(_judgment(), {
        "is_refusal": True,
        "factual_all": "unsupported",
        "min_factual_claims": 1,
        "citation_verdicts": ["supported"],
        "correctness_range": [0.9, 1.0],
    })

    assert any("is_refusal" in item for item in errors)
    assert any("citation_verdicts" in item for item in errors)
    assert any("answer_correctness" in item for item in errors)


def test_calibration_can_require_verdict_without_brittle_claim_order():
    errors = calibration_errors(_judgment(), {
        "is_refusal": False,
        "factual_must_include": ["supported", "unsupported"],
        "citation_verdicts": ["partially_supported"],
        "correctness_range": [0.0, 1.0],
    })

    assert errors == []


def test_payload_excludes_unneeded_evidence_fields():
    payload = build_judge_payload(
        case={"question": "上限？", "expect_answer": ["400"]},
        answer="400 元", evidence=[{
            "evidence_id": "E1.1", "source_name": "制度", "location": "第 2 条",
            "snippet": "上限 400 元", "score": 0.99,
        }], submitted_claims=[{"text": "上限 400 元", "evidence_ids": ["E1.1"]}],
    )

    assert "score" not in payload["evidence"][0]
    assert "grading_note" not in payload["reference"]
    assert payload["submitted_claims"][0]["claim_index"] == 0


class _FakeCompletions:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="```json\n" + json.dumps(self.raw, ensure_ascii=False) + "\n```"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def test_judge_validates_claim_coverage_and_uses_cache(tmp_path):
    raw = _judgment().model_dump()
    completions = _FakeCompletions(raw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    cache = JsonlJudgeCache(tmp_path / "judge.jsonl")
    judge = LLMAnswerJudge(
        model="judge", base_url="https://invalid", api_key="x",
        timeout_seconds=1, cache=cache, client=client,
    )
    payload = build_judge_payload(
        case={"question": "问题"}, answer="答案", evidence=[],
        submitted_claims=[{"text": "结论", "evidence_ids": []}],
    )

    first, first_meta = judge.judge(payload)
    second, second_meta = judge.judge(payload)

    assert first.answer_correctness == second.answer_correctness == 0.75
    assert first_meta["cached"] is False
    assert first_meta["usage"] == {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }
    assert first_meta["attempt"] == 1
    assert first_meta["latency_ms"] >= 0
    assert second_meta["cached"] is True
    assert second_meta["latency_ms"] == 0
    assert completions.calls == 1


def test_judge_cache_key_changes_when_rubric_changes():
    payload = {"answer": "相同答案"}

    first = JsonlJudgeCache.key("judge", payload, rubric="规则一")
    second = JsonlJudgeCache.key("judge", payload, rubric="规则二")

    assert first != second


def test_judge_rejects_missing_citation_assessment():
    raw = _judgment(citation_judgments=[]).model_dump()
    completions = _FakeCompletions(raw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    judge = LLMAnswerJudge(
        model="judge", base_url="https://invalid", api_key="x",
        timeout_seconds=1, max_retries=0, client=client,
    )
    payload = build_judge_payload(
        case={"question": "问题"}, answer="答案", evidence=[],
        submitted_claims=[{"text": "结论", "evidence_ids": []}],
    )

    with pytest.raises(RuntimeError, match="连续校验失败"):
        judge.judge(payload)
