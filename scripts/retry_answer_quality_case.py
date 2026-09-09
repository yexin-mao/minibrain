"""Retry one answer-quality case without rebuilding its index between attempts."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from eval_answer_quality import (  # noqa: E402
    CACHE_PATH, ROOT, deterministic_answer_ok, ingest, load_cases, run_calibration,
    summarize,
)
from minibrain import identity  # noqa: E402
from minibrain.agent.graph import answer  # noqa: E402
from minibrain.config import get_config  # noqa: E402
from minibrain.db import close_all  # noqa: E402
from minibrain.evaluation.answer_quality import (  # noqa: E402
    JsonlJudgeCache, LLMAnswerJudge, build_judge_payload, judge_independence,
    quality_scores,
)
from minibrain.evaluation.provenance import build_provenance  # noqa: E402
from minibrain.observability import core as observability  # noqa: E402
from minibrain.scripts_purge import purge_user  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    args = parser.parse_args()
    if args.attempts <= 0 or args.retry_delay < 0:
        parser.error("attempts 必须为正数，retry-delay 不能为负数")

    cases = {case["id"]: case for case in load_cases("grounding")}
    if args.case_id not in cases:
        parser.error(f"未知用例 {args.case_id}")
    case = cases[args.case_id]
    cfg = get_config()
    judge_model = os.environ.get("JUDGE_MODEL") or cfg.agent_model
    independence = judge_independence(cfg.agent_model, judge_model)
    if not independence["is_independent"]:
        parser.error("补跑要求独立 Judge")
    judge = LLMAnswerJudge(
        model=judge_model,
        base_url=os.environ.get("JUDGE_BASE_URL") or cfg.agent_base_url,
        api_key=os.environ.get("JUDGE_API_KEY") or cfg.agent_api_key,
        timeout_seconds=cfg.llm_timeout_seconds,
        max_retries=1,
        cache=JsonlJudgeCache(CACHE_PATH),
    )
    calibration = run_calibration(judge)
    if any(not row["passed"] for row in calibration):
        raise RuntimeError("Judge 校准失败")

    user = identity.create_user(
        f"answer_retry_{uuid.uuid4().hex[:6]}", "pw123456", is_admin=False)
    attempt_errors = []
    row = None
    try:
        n_doc, n_tbl = ingest(user)
        print(f"语料 {n_doc} 篇 + {n_tbl} 张表 | {case['id']} | 最多 {args.attempts} 次")
        for attempt in range(1, args.attempts + 1):
            try:
                started = time.perf_counter()
                result = answer(user, case["question"])
                wall_latency_ms = round((time.perf_counter() - started) * 1000)
                run = observability.get_run(user, result.run_id) if result.run_id else {}
                evidence = [asdict(item) for item in result.evidence]
                claims = [asdict(item) for item in result.claims]
                payload = build_judge_payload(
                    case=case, answer=result.answer, evidence=evidence,
                    submitted_claims=claims)
                judgment, judge_meta = judge.judge(payload)
                scores = quality_scores(
                    judgment, expected_refusal=case.get("expect_refusal"),
                    submitted_claim_count=len(claims))
                row = {
                    "id": case["id"], "group": case["group"],
                    "question": case["question"], "reference": payload["reference"],
                    "answer": result.answer,
                    "deterministic_answer_ok": deterministic_answer_ok(case, result.answer),
                    "evidence": evidence, "submitted_claims": claims,
                    "deterministic_citation_metrics": asdict(result.citation_metrics),
                    "judgment": judgment.model_dump(), "scores": scores,
                    "generation": {
                        "run_id": result.run_id,
                        "latency_ms": run.get("latency_ms", wall_latency_ms),
                        "usage": {
                            "input_tokens": int(run.get("input_tokens", 0) or 0),
                            "output_tokens": int(run.get("output_tokens", 0) or 0),
                            "total_tokens": int(run.get("total_tokens", 0) or 0),
                        },
                        "forced_final": result.forced_final,
                    },
                    "trace": [asdict(item) for item in result.trace],
                    "judge": judge_meta, "error": None,
                    "retry_attempt": attempt,
                }
                print(f"第 {attempt} 次成功，correct={scores['answer_correctness']:.2f}")
                break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                attempt_errors.append(error)
                print(f"第 {attempt} 次失败：{error}", flush=True)
                if attempt < args.attempts:
                    time.sleep(args.retry_delay)

        rows = [row] if row else []
        report = {
            "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": "grounding-retry", "answer_model": cfg.agent_model,
            "judge_model": judge_model, "judge_is_same_model": False,
            "judge_independence": independence,
            "provenance": build_provenance(ROOT, scope="answer_quality", config={
                "answer_model": cfg.agent_model, "judge_model": judge_model,
                "agent_temperature": cfg.agent_temperature,
                "agent_max_steps": cfg.agent_max_steps,
                "evidence_token_budget": cfg.agent_evidence_token_budget,
                "evidence_limit": cfg.agent_evidence_limit,
                "context_candidate_pool": cfg.agent_context_candidate_pool,
                "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap,
                "parent_chunk_size": cfg.parent_chunk_size,
                "expand_parent_context": cfg.expand_parent_context,
            }),
            "calibration": calibration,
            "retry_attempt_errors": attempt_errors,
            "summary": summarize(rows), "rows": rows,
        }
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if row else 1
    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
