"""Aggregate repeated answer-quality reports without calling any model."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import defaultdict

METRICS = (
    "answer_correctness", "faithfulness", "citation_correctness",
    "answer_relevance", "refusal_accuracy", "generation_latency_p50_ms",
    "generation_latency_p95_ms",
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(reports: list[tuple[pathlib.Path, dict]]) -> dict:
    fingerprints = {
        report.get("provenance", {}).get("source_fingerprint_sha256")
        for _, report in reports
    }
    per_case: dict[str, list[dict]] = defaultdict(list)
    for path, report in reports:
        for row in report.get("rows", []):
            per_case[row["id"]].append({
                "report": path.name,
                "error": row.get("error"),
                "answer_correctness": row.get("scores", {}).get("answer_correctness"),
                "faithfulness": row.get("scores", {}).get("faithfulness"),
                "citation_correctness": row.get("scores", {}).get("citation_correctness"),
            })

    expected_cases = max((report.get("summary", {}).get("cases", 0)
                          for _, report in reports), default=0)
    completed_runs = sum(
        report.get("status") == "completed"
        and report.get("summary", {}).get("succeeded") == expected_cases
        for _, report in reports
    )
    metric_summary = {}
    for metric in METRICS:
        values = [
            report.get("summary", {}).get(metric)
            for _, report in reports
            if report.get("summary", {}).get(metric) is not None
        ]
        if values:
            metric_summary[metric] = _summary(values)

    cases = []
    for case_id, rows in sorted(per_case.items()):
        scored = [row["answer_correctness"] for row in rows
                  if row["answer_correctness"] is not None]
        cases.append({
            "id": case_id,
            "observations": len(rows),
            "errors": sum(row["error"] is not None for row in rows),
            "full_passes": sum(value == 1.0 for value in scored),
            "answer_correctness_mean": statistics.mean(scored) if scored else None,
            "rows": rows,
        })

    return {
        "status": "completed" if completed_runs == len(reports) else "incomplete",
        "runs": len(reports),
        "completed_runs": completed_runs,
        "expected_cases_per_run": expected_cases,
        "source_fingerprints": sorted(value for value in fingerprints if value),
        "consistent_source_fingerprint": len(fingerprints) == 1 and None not in fingerprints,
        "metrics": metric_summary,
        "unstable_or_failing_cases": [
            case for case in cases
            if case["errors"] or case["full_passes"] < len(reports)
        ],
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=pathlib.Path, nargs="+")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reports = [(path, json.loads(path.read_text(encoding="utf-8")))
               for path in args.reports]
    result = aggregate(reports)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "completed_runs": result["completed_runs"],
        "metrics": result["metrics"],
        "unstable_or_failing_cases": [
            {key: row[key] for key in (
                "id", "observations", "errors", "full_passes",
                "answer_correctness_mean")}
            for row in result["unstable_or_failing_cases"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
