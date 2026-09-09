"""Calibrate a conservative retrieval-level no-answer rule.

This is deliberately a labelled, held-out experiment rather than a magic cosine
constant.  It measures only whether retrieval found plausible evidence; relevant
evidence can still omit the requested attribute, so claim-level grounding remains
the final safety boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import uuid
from dataclasses import asdict

sys.path.insert(0, "src")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES = ROOT / "eval" / "results" / "retrieval-confidence-samples.json"
DEFAULT_REPORT = ROOT / "eval" / "results" / "retrieval-confidence-calibration.json"


def _validation(case_id: str) -> bool:
    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 4 == 0


def _score_rule(row: dict, threshold: float) -> bool:
    signals = row["signals"]
    dense = signals.get("dense_top_score")
    keyword = signals.get("keyword_top_score")
    return dense is not None and dense < threshold and (keyword is None or keyword <= 0)


def _metrics(rows: list[dict], threshold: float) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    no_answer = [row for row in rows if not row["answerable"]]
    false_rejects = [row["id"] for row in answerable if _score_rule(row, threshold)]
    caught = [row["id"] for row in no_answer if _score_rule(row, threshold)]
    return {
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "false_rejects": false_rejects,
        "false_reject_rate": len(false_rejects) / len(answerable) if answerable else 0.0,
        "no_answer_caught": caught,
        "no_answer_recall": len(caught) / len(no_answer) if no_answer else 0.0,
    }


def calibrate(rows: list[dict]) -> dict:
    calibration = [row for row in rows if row["split"] == "calibration"]
    validation = [row for row in rows if row["split"] == "validation"]
    scores = sorted({
        float(row["signals"]["dense_top_score"])
        for row in calibration if row["signals"].get("dense_top_score") is not None
    })
    candidates = [scores[0] - 1e-6] if scores else [0.0]
    candidates += [(left + right) / 2 for left, right in zip(scores, scores[1:])]
    candidates += [scores[-1] + 1e-6] if scores else []

    safe = []
    for threshold in candidates:
        metric = _metrics(calibration, threshold)
        if not metric["false_rejects"]:
            safe.append((len(metric["no_answer_caught"]), threshold, metric))
    _, threshold, calibration_metrics = max(safe, key=lambda item: (item[0], item[1]))
    validation_metrics = _metrics(validation, threshold)
    # A hard score gate needs both zero held-out false rejects and at least two
    # held-out negative examples with a non-zero catch. Otherwise it stays report-only.
    enabled = (
        not validation_metrics["false_rejects"]
        and validation_metrics["no_answer"] >= 2
        and bool(validation_metrics["no_answer_caught"])
    )
    return {
        "version": "v1-dense-bm25-heldout",
        "score_gate_enabled": enabled,
        "dense_no_evidence_below": round(threshold, 6),
        "rule": "dense_top < threshold AND bm25 has no positive hit",
        "split": "sha256(case_id) mod 4 == 0 is held-out validation",
        "calibration": calibration_metrics,
        "validation": validation_metrics,
        "interpretation": (
            "held-out constraints passed; score rule may be enforced"
            if enabled else
            "score separation is insufficient; keep score rule report-only"
        ),
    }


def _cases() -> list[dict]:
    positives = []
    for filename in ("probes.json", "probes_blindspot.json"):
        for case in json.loads((ROOT / "eval" / filename).read_text(encoding="utf-8")):
            positives.append({
                "id": case["id"], "question": case["question"],
                "answerable": True, "kind": case.get("category", "answerable"),
            })
    negatives = [{
        "id": case["id"], "question": case["question"],
        "answerable": False, "kind": case.get("kind", "no-answer"),
    } for case in json.loads(
        (ROOT / "eval" / "probes_hallucination.json").read_text(encoding="utf-8"))
        if case.get("expect_refusal") is True]
    return positives + negatives


def collect_samples() -> list[dict]:
    from minibrain import gateway, identity
    from minibrain.config import get_config
    from minibrain.db import close_all
    from minibrain.scripts_purge import purge_user

    cfg = get_config()
    if not cfg.embedding_configured:
        raise RuntimeError("EMBEDDING_API_KEY 未配置，不能采集检索信号")
    user = identity.create_user(
        f"confidence_{uuid.uuid4().hex[:6]}", "pw123456", is_admin=False)
    try:
        for path in sorted((ROOT / "eval" / "corpus").glob("*.md")):
            document_id = gateway.call(
                "vector-rag", "upload_document", user, None,
                path.name, path.read_text(encoding="utf-8"),
            )
            gateway.process("vector-rag", document_id)

        rows = []
        for case in _cases():
            result = gateway.search(
                "vector-rag", user, case["question"], top_k=5, explain=True)
            rows.append({
                **case,
                "split": "validation" if _validation(case["id"]) else "calibration",
                "signals": asdict(result.retrieval_signals),
                "returned_files": list(dict.fromkeys(
                    item.location.split(" #", 1)[0] for item in result.evidence)),
            })
        return rows
    finally:
        purge_user(user)
        close_all()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path,
                        help="只重算已有 samples，不调用 embedding")
    parser.add_argument("--samples", type=pathlib.Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        rows = payload["rows"] if isinstance(payload, dict) else payload
    else:
        rows = collect_samples()
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        args.samples.write_text(
            json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = calibrate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
