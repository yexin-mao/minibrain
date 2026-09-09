"""独立 Judge 与人工盲标的一致率。

模板刻意不包含 Judge 判定，避免标注者被模型答案锚定。分析时再用报告摘要校验
模板来源，并按稳定的 case / claim 下标对齐。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Hashable

VERDICTS = {
    "supported", "partially_supported", "unsupported", "not_applicable",
}


def report_digest(report: dict[str, Any]) -> str:
    raw = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_human_annotation_template(report: dict[str, Any]) -> dict[str, Any]:
    """从 Judge 报告生成盲标文件；不复制 judgment / scores。"""
    rows = []
    for row in report.get("rows", []):
        if row.get("error") is not None:
            continue
        judgment = row.get("judgment", {})
        rows.append({
            "id": row["id"],
            "question": row.get("question"),
            "answer": row.get("answer"),
            "reference": row.get("reference"),
            "evidence": row.get("evidence", []),
            "factual_claims": [{
                "claim_index": index,
                "claim": claim.get("claim", ""),
                "human_verdict": None,
            } for index, claim in enumerate(judgment.get("factual_claims", []))],
            "submitted_claims": [{
                "claim_index": index,
                "text": claim.get("text", ""),
                "evidence_ids": claim.get("evidence_ids", []),
                "human_verdict": None,
            } for index, claim in enumerate(row.get("submitted_claims", []))],
            "human": {
                "answer_correctness": None,
                "answer_relevance": None,
                "is_refusal": None,
                "note": "",
            },
        })
    return {
        "schema_version": "human-answer-quality-v1",
        "source_report_sha256": report_digest(report),
        "answer_model": report.get("answer_model"),
        "judge_model": report.get("judge_model"),
        "annotator": "",
        "instructions": {
            "scores": "answer_correctness / answer_relevance 填 0 到 1",
            "is_refusal": "明确表示当前证据无法确定目标事实时填 true",
            "verdicts": sorted(VERDICTS),
            "blindness": "不要查看源报告中的 judgment / scores 后再标注",
            "factual_scope": "factual_claims 由 Judge 抽取；这里只盲标其支持状态，不测 claim 抽取完整率",
        },
        "rows": rows,
    }


def cohen_kappa(pairs: list[tuple[Hashable, Hashable]]) -> float | None:
    """无第三方依赖的 Cohen's κ；常量标签导致分母为零时返回 None。"""
    if not pairs:
        return None
    labels = {value for pair in pairs for value in pair}
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[label] / total * right_counts[label] / total
        for label in labels
    )
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 6)


def _categorical(pairs: list[tuple[Hashable, Hashable]]) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "exact_agreement": (
            round(sum(left == right for left, right in pairs) / len(pairs), 6)
            if pairs else None
        ),
        "cohen_kappa": cohen_kappa(pairs),
    }


def _continuous(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    differences = [abs(left - right) for left, right in pairs]
    return {
        "n": len(pairs),
        "mae": round(sum(differences) / len(differences), 6) if pairs else None,
        "within_0_2": (
            round(sum(value <= 0.2 for value in differences) / len(pairs), 6)
            if pairs else None
        ),
    }


def _score(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是 0 到 1 的数字")
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{field} 必须在 0 到 1 之间")
    return number


def _verdict(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if value not in VERDICTS:
        raise ValueError(f"{field} 不是合法 verdict：{value}")
    return str(value)


def judge_human_agreement(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    if labels.get("schema_version") != "human-answer-quality-v1":
        raise ValueError("未知的人工标注 schema_version")
    if labels.get("source_report_sha256") != report_digest(report):
        raise ValueError("人工标注与 Judge 报告摘要不匹配，不能进行对齐")

    report_rows = {
        row["id"]: row for row in report.get("rows", [])
        if row.get("error") is None
    }
    label_rows = {row["id"]: row for row in labels.get("rows", [])}
    unknown = sorted(set(label_rows) - set(report_rows))
    if unknown:
        raise ValueError(f"人工标注包含未知 case：{unknown}")

    refusal_pairs: list[tuple[bool, bool]] = []
    factual_pairs: list[tuple[str, str]] = []
    citation_pairs: list[tuple[str, str]] = []
    correctness_pairs: list[tuple[float, float]] = []
    relevance_pairs: list[tuple[float, float]] = []
    labelled_cases = 0

    for case_id, human_row in label_rows.items():
        judged = report_rows[case_id]
        judgment = judged["judgment"]
        human = human_row.get("human", {})
        case_has_label = False

        human_refusal = human.get("is_refusal")
        if human_refusal is not None:
            if not isinstance(human_refusal, bool):
                raise ValueError(f"{case_id}.is_refusal 必须是 boolean")
            refusal_pairs.append((bool(judgment["is_refusal"]), human_refusal))
            case_has_label = True

        for field, target in (
            ("answer_correctness", correctness_pairs),
            ("answer_relevance", relevance_pairs),
        ):
            human_score = _score(human.get(field), f"{case_id}.{field}")
            if human_score is not None:
                target.append((float(judgment[field]), human_score))
                case_has_label = True

        judged_factual = judgment.get("factual_claims", [])
        human_factual = human_row.get("factual_claims", [])
        if len(human_factual) != len(judged_factual):
            raise ValueError(f"{case_id}.factual_claims 数量已变化")
        for index, item in enumerate(human_factual):
            value = _verdict(
                item.get("human_verdict"), f"{case_id}.factual_claims[{index}]")
            if value is not None:
                factual_pairs.append((judged_factual[index]["verdict"], value))
                case_has_label = True

        judged_citations = {
            int(item["claim_index"]): item
            for item in judgment.get("citation_judgments", [])
        }
        human_citations = human_row.get("submitted_claims", [])
        if len(human_citations) != len(judged_citations):
            raise ValueError(f"{case_id}.submitted_claims 数量已变化")
        for item in human_citations:
            index = int(item["claim_index"])
            value = _verdict(
                item.get("human_verdict"), f"{case_id}.submitted_claims[{index}]")
            if value is not None:
                citation_pairs.append((judged_citations[index]["verdict"], value))
                case_has_label = True

        labelled_cases += int(case_has_label)

    return {
        "schema_version": "judge-human-agreement-v1",
        "answer_model": report.get("answer_model"),
        "judge_model": report.get("judge_model"),
        "judge_is_same_model": report.get("judge_is_same_model"),
        "independence_check": "exact_model_name_only",
        "limitations": [
            "factual_verdict 只衡量给定 claim 的判定一致率，不衡量 Judge 是否漏抽事实",
            "模型名不同不能排除同供应商、同系列或别名关系",
        ],
        "annotator": labels.get("annotator", ""),
        "coverage": {
            "eligible_cases": len(report_rows),
            "labelled_cases": labelled_cases,
        },
        "is_refusal": _categorical(refusal_pairs),
        "factual_verdict": _categorical(factual_pairs),
        "citation_verdict": _categorical(citation_pairs),
        "answer_correctness": _continuous(correctness_pairs),
        "answer_relevance": _continuous(relevance_pairs),
    }
