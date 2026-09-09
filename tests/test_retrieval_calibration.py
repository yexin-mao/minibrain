from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from calibrate_retrieval_confidence import calibrate  # noqa: E402


def _row(case_id: str, answerable: bool, dense: float, *, validation=False):
    return {
        "id": case_id, "answerable": answerable,
        "split": "validation" if validation else "calibration",
        "signals": {"dense_top_score": dense, "keyword_top_score": None},
    }


def test_calibration_enables_only_when_heldout_has_no_false_rejects():
    rows = [
        _row("p1", True, 0.8), _row("n1", False, 0.2),
        _row("p2", True, 0.75, validation=True),
        _row("n2", False, 0.1, validation=True),
        _row("n3", False, 0.3, validation=True),
    ]
    report = calibrate(rows)
    assert report["score_gate_enabled"] is True
    assert report["validation"]["false_rejects"] == []


def test_calibration_keeps_gate_report_only_when_validation_would_be_rejected():
    rows = [
        _row("p1", True, 0.8), _row("n1", False, 0.5),
        _row("p2", True, 0.2, validation=True),
        _row("n2", False, 0.1, validation=True),
        _row("n3", False, 0.3, validation=True),
    ]
    report = calibrate(rows)
    assert report["score_gate_enabled"] is False
    assert report["validation"]["false_rejects"] == ["p2"]
