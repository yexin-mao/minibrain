from copy import deepcopy

import pytest

from minibrain.evaluation.agreement import (
    build_human_annotation_template,
    cohen_kappa,
    judge_human_agreement,
)
from minibrain.evaluation.answer_quality import judge_independence


def _report():
    def row(case_id, refusal, correctness, verdict):
        return {
            "id": case_id, "question": f"问题 {case_id}", "answer": "答案",
            "reference": {}, "evidence": [], "error": None,
            "submitted_claims": [{"text": "结论", "evidence_ids": ["E1.1"]}],
            "judgment": {
                "answer_correctness": correctness,
                "answer_relevance": 1.0,
                "is_refusal": refusal,
                "factual_claims": [{"claim": "结论", "verdict": verdict}],
                "citation_judgments": [{"claim_index": 0, "verdict": verdict}],
            },
            "scores": {},
        }
    return {
        "status": "completed", "answer_model": "answer-a",
        "judge_model": "judge-b", "judge_is_same_model": False,
        "rows": [
            row("c1", False, 0.8, "supported"),
            row("c2", True, 0.2, "unsupported"),
        ],
    }


def test_template_is_blind_and_keeps_alignment_fields():
    template = build_human_annotation_template(_report())

    assert "judgment" not in template["rows"][0]
    assert "scores" not in template["rows"][0]
    assert template["rows"][0]["factual_claims"][0] == {
        "claim_index": 0, "claim": "结论", "human_verdict": None,
    }
    assert template["rows"][0]["submitted_claims"][0]["human_verdict"] is None


def test_agreement_reports_kappa_exact_rate_and_continuous_error():
    report = _report()
    labels = build_human_annotation_template(report)
    labels["annotator"] = "reviewer-1"
    first, second = labels["rows"]
    first["human"].update({
        "answer_correctness": 1.0, "answer_relevance": 1.0,
        "is_refusal": False,
    })
    second["human"].update({
        "answer_correctness": 0.4, "answer_relevance": 0.8,
        "is_refusal": False,
    })
    first["factual_claims"][0]["human_verdict"] = "supported"
    second["factual_claims"][0]["human_verdict"] = "unsupported"
    first["submitted_claims"][0]["human_verdict"] = "supported"
    second["submitted_claims"][0]["human_verdict"] = "supported"

    result = judge_human_agreement(report, labels)

    assert result["coverage"] == {"eligible_cases": 2, "labelled_cases": 2}
    assert result["is_refusal"] == {
        "n": 2, "exact_agreement": 0.5, "cohen_kappa": 0.0,
    }
    assert result["factual_verdict"]["exact_agreement"] == 1.0
    assert result["factual_verdict"]["cohen_kappa"] == 1.0
    assert result["citation_verdict"]["exact_agreement"] == 0.5
    assert result["answer_correctness"]["mae"] == 0.2
    assert result["answer_correctness"]["within_0_2"] == 1.0


def test_digest_prevents_comparing_labels_to_changed_report():
    report = _report()
    labels = build_human_annotation_template(report)
    changed = deepcopy(report)
    changed["rows"][0]["answer"] = "修改后的答案"

    with pytest.raises(ValueError, match="摘要不匹配"):
        judge_human_agreement(changed, labels)


def test_invalid_human_score_is_rejected():
    report = _report()
    labels = build_human_annotation_template(report)
    labels["rows"][0]["human"]["answer_correctness"] = 1.2

    with pytest.raises(ValueError, match="0 到 1"):
        judge_human_agreement(report, labels)


def test_kappa_returns_none_for_constant_identical_labels():
    assert cohen_kappa([("x", "x"), ("x", "x")]) is None


def test_independent_judge_check_is_honest_about_its_limit():
    same = judge_independence("DeepSeek/V4", " deepseek/v4 ")
    different = judge_independence("answer-model", "judge-model")

    assert same["is_independent"] is False
    assert different["is_independent"] is True
    assert "别名" in different["limitation"]
