"""Fail CI when a NanoBEIR retrieval report regresses beyond the agreed budget.

The baseline is a small, frozen measurement artifact.  The candidate is normally
produced by ``eval_nanobeir.py``.  Keeping this checker API-free lets CI enforce
the decision deterministically; refreshing the candidate remains an explicit,
cost-bearing benchmark step.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import math
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "eval" / "baselines" / "nanobeir-hybrid.json"
DEFAULT_CANDIDATE = ROOT / "eval" / "results" / "nanobeir-global-bm25.json"
QUALITY_METRICS = ("mrr", "recall@5", "complete_recall@5", "ndcg@10")
REQUIRED_METRICS = (
    "mrr", "recall@3", "recall@5", "recall@10",
    "precision@3", "precision@5", "precision@10",
    "hit@3", "hit@5", "hit@10",
    "complete_recall@3", "complete_recall@5", "complete_recall@10",
    "ndcg@3", "ndcg@5", "ndcg@10",
)


def _load(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取评测报告 {path}: {exc}") from exc


def compare_reports(
    baseline: dict,
    candidate: dict,
    *,
    arm: str = "hybrid",
    max_quality_drop: float = 0.03,
    max_latency_increase: float = 0.50,
) -> list[str]:
    """Return human-readable violations; an empty list means the gate passes."""
    failures: list[str] = []
    baseline_tasks = baseline.get("tasks", {})
    candidate_tasks = candidate.get("tasks", {})
    if not baseline_tasks:
        return ["baseline 不包含 tasks"]

    for task_name, baseline_task in baseline_tasks.items():
        try:
            old_arm = baseline_task["arms"][arm]
            new_arm = candidate_tasks[task_name]["arms"][arm]
        except (KeyError, TypeError):
            failures.append(f"{task_name}/{arm}: candidate 缺少对应评测结果")
            continue

        old_metrics = old_arm.get("metrics", {})
        new_metrics = new_arm.get("metrics", {})
        for metric in QUALITY_METRICS:
            if metric not in old_metrics or metric not in new_metrics:
                failures.append(f"{task_name}/{arm}/{metric}: 指标缺失")
                continue
            old = float(old_metrics[metric])
            new = float(new_metrics[metric])
            if new < old - max_quality_drop:
                failures.append(
                    f"{task_name}/{arm}/{metric}: {old:.4f} -> {new:.4f} "
                    f"(下降 {old - new:.4f} > {max_quality_drop:.4f})"
                )

        latency_key = "latency_median_ms_without_query_embedding"
        old_latency = old_arm.get(latency_key)
        new_latency = new_arm.get(latency_key)
        if old_latency is not None and new_latency is not None:
            limit = float(old_latency) * (1 + max_latency_increase)
            if float(new_latency) > limit:
                failures.append(
                    f"{task_name}/{arm}/p50: {float(old_latency):.1f}ms -> "
                    f"{float(new_latency):.1f}ms (上限 {limit:.1f}ms)"
                )
    return failures


def validate_candidate(candidate: dict, *, expected_fingerprint: str) -> list[str]:
    """Reject stale, partial, malformed, or non-standard benchmark artifacts."""
    failures: list[str] = []
    provenance = candidate.get("provenance", {})
    if provenance.get("source_fingerprint_sha256") != expected_fingerprint:
        failures.append("candidate 与当前源码指纹不一致，请重跑 eval_nanobeir.py")
    if not provenance.get("generated_at"):
        failures.append("candidate 缺少 provenance.generated_at")

    tasks = candidate.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        return [*failures, "candidate 不包含 tasks"]
    for task_name, task in tasks.items():
        if task.get("standard_run") is not True:
            failures.append(f"{task_name}: 不是完整 standard_run")
        queries = task.get("queries")
        evaluated = task.get("evaluated_queries")
        if not isinstance(queries, int) or queries <= 0 or evaluated != queries:
            failures.append(f"{task_name}: evaluated_queries 必须等于 queries")
        if not task.get("repo_id") or not task.get("revision"):
            failures.append(f"{task_name}: 缺少数据集 repo_id/revision")
        arm = task.get("arms", {}).get("hybrid", {})
        metrics = arm.get("metrics", {})
        for metric in REQUIRED_METRICS:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f"{task_name}/hybrid/{metric}: 指标缺失或非法")
            elif not 0 <= float(value) <= 1:
                failures.append(f"{task_name}/hybrid/{metric}: 指标超出 [0, 1]")
        for latency in (
            "latency_median_ms_without_query_embedding",
            "latency_p95_ms_without_query_embedding",
        ):
            value = arm.get(latency)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                failures.append(f"{task_name}/hybrid/{latency}: 延迟缺失或非法")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=pathlib.Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--arm", default="hybrid")
    parser.add_argument("--max-quality-drop", type=float, default=0.03)
    parser.add_argument("--max-latency-increase", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        baseline = _load(args.baseline)
        candidate = _load(args.candidate)
        from minibrain.evaluation.provenance import source_fingerprint

        failures = validate_candidate(
            candidate, expected_fingerprint=source_fingerprint(ROOT))
        failures += compare_reports(
            baseline,
            candidate,
            arm=args.arm,
            max_quality_drop=args.max_quality_drop,
            max_latency_increase=args.max_latency_increase,
        )
    except ValueError as exc:
        print(f"评测门禁配置错误: {exc}", file=sys.stderr)
        return 2

    if failures:
        print("RAG 检索回归门禁失败:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"RAG 检索回归门禁通过：{len(baseline['tasks'])} 个任务，arm={args.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
