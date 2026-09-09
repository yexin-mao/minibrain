from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from check_eval_regression import compare_reports, validate_candidate  # noqa: E402


def _report(*, mrr: float = 0.8, latency: float = 100.0) -> dict:
    return {
        "tasks": {
            "tiny": {
                "arms": {
                    "hybrid": {
                        "metrics": {
                            "mrr": mrr,
                            "recall@5": 0.8,
                            "complete_recall@5": 0.8,
                            "ndcg@10": 0.8,
                        },
                        "latency_median_ms_without_query_embedding": latency,
                    }
                }
            }
        }
    }


def test_eval_gate_accepts_measurement_within_budget():
    assert compare_reports(_report(), _report(mrr=0.77, latency=150.0)) == []


def test_eval_gate_rejects_quality_regression():
    failures = compare_reports(_report(), _report(mrr=0.75))
    assert any("mrr" in failure and "下降" in failure for failure in failures)


def test_eval_gate_rejects_latency_regression():
    failures = compare_reports(_report(), _report(latency=151.0))
    assert any("p50" in failure and "上限" in failure for failure in failures)


def test_eval_gate_rejects_missing_task():
    failures = compare_reports(_report(), {"tasks": {}})
    assert failures == ["tiny/hybrid: candidate 缺少对应评测结果"]


def _complete_candidate() -> dict:
    metrics = {name: 0.8 for name in (
        "mrr", "recall@3", "recall@5", "recall@10",
        "precision@3", "precision@5", "precision@10",
        "hit@3", "hit@5", "hit@10",
        "complete_recall@3", "complete_recall@5", "complete_recall@10",
        "ndcg@3", "ndcg@5", "ndcg@10",
    )}
    return {
        "provenance": {
            "generated_at": "2026-08-25T00:00:00+00:00",
            "source_fingerprint_sha256": "current",
        },
        "tasks": {"tiny": {
            "standard_run": True, "queries": 50, "evaluated_queries": 50,
            "repo_id": "example/tiny", "revision": "abc123",
            "arms": {"hybrid": {
                "metrics": metrics,
                "latency_median_ms_without_query_embedding": 10.0,
                "latency_p95_ms_without_query_embedding": 15.0,
            }},
        }},
    }


def test_candidate_validation_accepts_complete_current_report():
    assert validate_candidate(_complete_candidate(), expected_fingerprint="current") == []


def test_candidate_validation_rejects_stale_or_partial_report():
    report = _complete_candidate()
    report["provenance"]["source_fingerprint_sha256"] = "old"
    report["tasks"]["tiny"]["standard_run"] = False
    report["tasks"]["tiny"]["arms"]["hybrid"].pop(
        "latency_p95_ms_without_query_embedding")
    failures = validate_candidate(report, expected_fingerprint="current")
    assert any("源码指纹" in failure for failure in failures)
    assert any("standard_run" in failure for failure in failures)
    assert any("p95" in failure for failure in failures)
