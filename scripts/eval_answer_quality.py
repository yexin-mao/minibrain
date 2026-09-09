"""端到端 RAG 回答质量评测：生成答案，再用结构化 LLM Judge 逐 claim 判定。

默认 smoke 套件只有 8 题；完整 grounding 套件必须显式传 --suite grounding。
Judge 有独立 JSONL 缓存，但答案生成每次都会重新执行，因此两部分成本分开记录。

跑法（会产生模型调用）：
  uv run --no-sync python scripts/eval_answer_quality.py
  uv run --no-sync python scripts/eval_answer_quality.py --suite grounding
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, "src")

from minibrain import gateway, identity  # noqa: E402
from minibrain.agent.graph import answer  # noqa: E402
from minibrain.config import get_config  # noqa: E402
from minibrain.db import close_all  # noqa: E402
from minibrain.evaluation.answer_quality import (  # noqa: E402
    JsonlJudgeCache,
    LLMAnswerJudge,
    build_judge_payload,
    calibration_errors,
    judge_independence,
    quality_scores,
)
from minibrain.evaluation.provenance import build_provenance  # noqa: E402
from minibrain.observability import core as observability  # noqa: E402
from minibrain.scripts_purge import purge_user  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_DOC = ROOT / "eval" / "corpus"
CORPUS_TBL = ROOT / "eval" / "corpus_table"
OUT_DIR = ROOT / "eval" / "results"
CACHE_PATH = ROOT / "eval" / "cache" / "answer-quality-judge.jsonl"
CALIBRATION_PATH = ROOT / "eval" / "judge_calibration.json"

SMOKE_IDS = {
    "hal-01", "hal-03", "hal-04", "hal-08", "hal-10", "hal-11",
    "mh-01", "mh-08",
}


def _load(path: str) -> list[dict[str, Any]]:
    return json.loads((ROOT / "eval" / path).read_text(encoding="utf-8"))


def load_cases(suite: str, limit: int | None = None) -> list[dict[str, Any]]:
    hallucination = [dict(row, group=f"幻觉·{row['kind']}")
                     for row in _load("probes_hallucination.json")]
    multihop = [dict(row, group="正常·多跳") for row in _load("probes_multihop.json")]
    normal = [dict(row, group=f"正常·{row['category']}") for row in _load("probes.json")
              if row["category"] in ("对照组·单点事实", "全局聚合")]
    cases = hallucination + multihop + normal
    if suite == "smoke":
        selected = [row for row in cases if row["id"] in SMOKE_IDS]
        # 数据集 ID 改名时不能静默少跑。
        missing = SMOKE_IDS - {row["id"] for row in selected}
        if missing:
            raise RuntimeError(f"smoke 用例不存在：{sorted(missing)}")
        cases = selected
    return cases[:limit] if limit else cases


def ingest(user) -> tuple[int, int]:
    docs = sorted(CORPUS_DOC.glob("*.md"))
    source = gateway.call(
        "vector-rag", "create_source", user,
        f"benchmark/answer-quality/{uuid.uuid4().hex[:8]}",
    )
    from minibrain.modules.vector_rag import chain
    chain.ingest_many_raw(
        [(path.name, path.read_text(encoding="utf-8")) for path in docs],
        source_name=str(source["name"]), visibility="private", owner_id=user.user_id,
    )
    tables = sorted(CORPUS_TBL.glob("*.csv"))
    for path in tables:
        dataset_id = gateway.call(
            "table-rag", "upload_csv", user, None, path.name, path.read_bytes())
        gateway.process("table-rag", dataset_id)
    return len(docs), len(tables)


def deterministic_answer_ok(case: dict[str, Any], text: str) -> bool | None:
    wanted = [re.sub(r"[,\s，、]", "", str(item))
              for item in case.get("expect_answer", [])]
    if not wanted:
        return None
    normalized = re.sub(r"[,\s，、]", "", text)
    hits = [item in normalized for item in wanted]
    return any(hits) if case.get("answer_match") == "any" else all(hits)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["scores"][key] for row in rows if row.get("scores", {}).get(key) is not None]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("error") is None]
    latencies = [row["generation"]["latency_ms"] for row in successful]
    return {
        "cases": len(rows),
        "succeeded": len(successful),
        "answer_correctness": _mean(successful, "answer_correctness"),
        "faithfulness": _mean(successful, "faithfulness"),
        "citation_correctness": _mean(successful, "citation_correctness"),
        "answer_relevance": _mean(successful, "answer_relevance"),
        "refusal_accuracy": _mean(successful, "refusal_correct"),
        "generation_latency_p50_ms": statistics.median(latencies) if latencies else None,
        "generation_latency_p95_ms": sorted(latencies)[
            max(0, int(len(latencies) * .95 + .999999) - 1)] if latencies else None,
        "generation_tokens": sum(row["generation"]["usage"]["total_tokens"]
                                 for row in successful),
        "judge_tokens_billed": sum(
            row["judge"]["usage"]["total_tokens"] for row in successful
            if not row["judge"]["cached"]),
        "judge_cache_hits": sum(row["judge"]["cached"] for row in successful),
    }


def _money(tokens_in: int, tokens_out: int, input_price: float | None,
           output_price: float | None) -> float | None:
    if input_price is None or output_price is None:
        return None
    return tokens_in / 1_000_000 * input_price + tokens_out / 1_000_000 * output_price


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """原子写入；长评测中断时至少保留已完成题目的明细。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_calibration(judge: LLMAnswerJudge) -> list[dict[str, Any]]:
    cases = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        judgment, meta = judge.judge(case["payload"])
        errors = calibration_errors(judgment, case["expected"])
        rows.append({
            "id": case["id"], "passed": not errors, "errors": errors,
            "judgment": judgment.model_dump(), "judge": meta,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "grounding"), default="smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append",
                        help="只跑指定用例，可重复传入；用于低成本回归单个失败样本")
    parser.add_argument("--output", type=pathlib.Path,
                        default=OUT_DIR / "answer-quality.json")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-input-price", type=float,
                        help="每百万 input token 的价格，仅用于报告估算")
    parser.add_argument("--judge-output-price", type=float,
                        help="每百万 output token 的价格，仅用于报告估算")
    parser.add_argument(
        "--require-independent-judge", action="store_true",
        help="Judge 与回答模型同名时中止；正式对外评测建议始终开启",
    )
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--rejudge", type=pathlib.Path,
                        help="复用已有报告中的答案/证据，只重跑 Judge，不入库和生成")
    args = parser.parse_args()

    cfg = get_config()
    judge_model = args.judge_model or os.environ.get("JUDGE_MODEL") or cfg.agent_model
    judge_base_url = os.environ.get("JUDGE_BASE_URL") or cfg.agent_base_url
    judge_api_key = os.environ.get("JUDGE_API_KEY") or cfg.agent_api_key
    if not cfg.agent_configured or judge_api_key.startswith("<"):
        print("AGENT_API_KEY / JUDGE_API_KEY 未配置", file=sys.stderr)
        return 1

    source_for_rejudge = (
        json.loads(args.rejudge.read_text(encoding="utf-8"))
        if args.rejudge else None
    )
    evaluated_answer_model = (
        source_for_rejudge.get("answer_model") if source_for_rejudge
        else cfg.agent_model
    )
    independence = judge_independence(evaluated_answer_model, judge_model)
    if args.require_independent_judge and not independence["is_independent"]:
        print(
            "Judge 与回答模型相同，已按 --require-independent-judge 中止。"
            "请配置不同的 JUDGE_MODEL。",
            file=sys.stderr,
        )
        return 3

    cases = load_cases(args.suite, args.limit)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            parser.error(f"当前 suite 中没有用例：{sorted(missing)}")
    cache = JsonlJudgeCache(CACHE_PATH)
    judge = LLMAnswerJudge(
        model=judge_model, base_url=judge_base_url, api_key=judge_api_key,
        timeout_seconds=cfg.llm_timeout_seconds, max_retries=1, cache=cache,
    )
    calibration = [] if args.skip_calibration else run_calibration(judge)
    failed_calibration = [row for row in calibration if not row["passed"]]
    if calibration:
        print(f"Judge 校准：{len(calibration) - len(failed_calibration)}/{len(calibration)} 通过")
    if failed_calibration:
        for row in failed_calibration:
            print(f"  {row['id']}: {'；'.join(row['errors'])}", file=sys.stderr)
        print("Judge 未通过人工校准，已中止正式评测。", file=sys.stderr)
        return 2
    if args.calibrate_only:
        return 0

    if args.rejudge:
        source = source_for_rejudge
        cases_by_id = {case["id"]: case for case in load_cases("grounding")}
        rows = []
        print(f"复用 {args.rejudge}，只重跑 Judge")
        for index, old_row in enumerate(source["rows"], start=1):
            if old_row.get("error") is not None:
                rows.append(old_row)
                continue
            case = cases_by_id[old_row["id"]]
            payload = build_judge_payload(
                case=case, answer=old_row["answer"], evidence=old_row["evidence"],
                submitted_claims=old_row["submitted_claims"],
            )
            judgment, judge_meta = judge.judge(payload)
            row = dict(old_row)
            row["reference"] = payload["reference"]
            row["judgment"] = judgment.model_dump()
            row["scores"] = quality_scores(
                judgment, expected_refusal=case.get("expect_refusal"),
                submitted_claim_count=len(old_row["submitted_claims"]),
            )
            row["judge"] = judge_meta
            rows.append(row)
            _write_json(args.output, {
                "status": "running", "suite": source.get("suite", "rejudge"),
                "answer_model": source.get("answer_model"),
                "judge_model": judge_model, "calibration": calibration, "rows": rows,
            })
            print(f"  {index:>2}/{len(source['rows'])} {row['id']:<9} "
                  f"correct={row['scores']['answer_correctness']:.2f}")
        summary = summarize(rows)
        report = {
            "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": source.get("suite", "rejudge"),
            "answer_model": source.get("answer_model"), "judge_model": judge_model,
            "judge_is_same_model": judge_model == source.get("answer_model"),
            "judge_independence": independence,
            "provenance": build_provenance(ROOT, scope="answer_quality", config={
                "answer_model": source.get("answer_model"),
                "judge_model": judge_model,
            }),
            "rejudged_from": str(args.rejudge), "calibration": calibration,
            "summary": summary, "rows": rows,
        }
        _write_json(args.output, report)
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"明细已写入 {args.output.resolve()}")
        return 0 if summary["succeeded"] == summary["cases"] else 1

    user = identity.create_user(
        f"answer_quality_{uuid.uuid4().hex[:6]}", "pw123456", is_admin=False)
    rows: list[dict[str, Any]] = []
    try:
        n_doc, n_tbl = ingest(user)
        print(f"语料 {n_doc} 篇 + {n_tbl} 张表 | {args.suite} {len(cases)} 题")
        print(f"回答模型 {cfg.agent_model} | Judge {judge_model}\n")
        for index, case in enumerate(cases, start=1):
            try:
                started = time.perf_counter()
                result = answer(user, case["question"])
                wall_latency_ms = round((time.perf_counter() - started) * 1000)
                run = observability.get_run(user, result.run_id) if result.run_id else {}
                evidence = [asdict(item) for item in result.evidence]
                claims = [asdict(item) for item in result.claims]
                payload = build_judge_payload(
                    case=case, answer=result.answer, evidence=evidence,
                    submitted_claims=claims,
                )
                judgment, judge_meta = judge.judge(payload)
                scores = quality_scores(
                    judgment, expected_refusal=case.get("expect_refusal"),
                    submitted_claim_count=len(claims),
                )
                rows.append({
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
                })
                print(f"  {index:>2}/{len(cases)} {case['id']:<9} "
                      f"correct={scores['answer_correctness']:.2f} "
                      f"faith={scores['faithfulness'] if scores['faithfulness'] is not None else '—'}")
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "id": case["id"], "group": case["group"],
                    "question": case["question"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  {index:>2}/{len(cases)} {case['id']:<9} ERROR {exc}")

            _write_json(args.output, {
                "status": "running", "suite": args.suite,
                "answer_model": cfg.agent_model, "judge_model": judge_model,
                "calibration": calibration, "rows": rows,
            })

        summary = summarize(rows)
        billable_judge_items = [
            row.get("judge", {}) for row in [*calibration, *rows]
            if row.get("judge") and not row["judge"].get("cached", False)
        ]
        judge_input = sum(item.get("usage", {}).get("input_tokens", 0)
                          for item in billable_judge_items)
        judge_output = sum(item.get("usage", {}).get("output_tokens", 0)
                           for item in billable_judge_items)
        calibration_tokens = sum(
            row.get("judge", {}).get("usage", {}).get("total_tokens", 0)
            for row in calibration if not row.get("judge", {}).get("cached", False))
        summary["calibration_judge_tokens_billed"] = calibration_tokens
        summary["total_judge_tokens_billed"] = judge_input + judge_output
        summary["estimated_judge_cost"] = _money(
            judge_input, judge_output, args.judge_input_price, args.judge_output_price)
        report = {
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": args.suite,
            "answer_model": cfg.agent_model,
            "judge_model": judge_model,
            "judge_is_same_model": judge_model == cfg.agent_model,
            "judge_independence": independence,
            "provenance": build_provenance(ROOT, scope="answer_quality", config={
                "answer_model": cfg.agent_model,
                "judge_model": judge_model,
                "agent_temperature": cfg.agent_temperature,
                "agent_max_steps": cfg.agent_max_steps,
                "evidence_token_budget": cfg.agent_evidence_token_budget,
                "evidence_limit": cfg.agent_evidence_limit,
                "context_candidate_pool": cfg.agent_context_candidate_pool,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "parent_chunk_size": cfg.parent_chunk_size,
                "expand_parent_context": cfg.expand_parent_context,
            }),
            "calibration": calibration,
            "summary": summary,
            "rows": rows,
        }
        _write_json(args.output, report)
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
        output_path = args.output.resolve()
        try:
            output_label = output_path.relative_to(ROOT)
        except ValueError:
            output_label = output_path
        print(f"明细已写入 {output_label}")
        if judge_model == cfg.agent_model:
            print("注意：当前是同模型自评；正式结论应换独立 Judge，并人工复核样本。")
        return 0 if summary["succeeded"] == summary["cases"] else 1
    finally:
        purge_user(user)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
