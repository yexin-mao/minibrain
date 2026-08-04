"""溯源检查。纯函数，不调模型也不碰数据库。

这个文件里的每条断言都对应一个**实际踩过或预见到的误判**——
溯源检查最大的风险不是漏报，是**误报**：把正确答案标成幻觉，
几次之后就没人信这个指标了。
"""

from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from grounding import (                       # noqa: E402
    check, extract_identifiers, extract_numbers, normalize_number,
)


# ---------------------------------------------------------------- 抽数字

def test_citation_markers_are_not_facts():
    """★ 引用标记 [1] [2] 必须先去掉，否则会被当成事实数字。"""
    assert extract_numbers("后端组共有 8 人 [1][2]") == ["8"]


def test_markdown_does_not_break_numbers():
    """**8 人** 这种加粗常见于模型输出。"""
    assert extract_numbers("共有 **32** 人") == ["32"]


def test_thousand_separators_are_normalised():
    """3,053,000 和 3053000 必须可比——模型两种写法都会用。"""
    assert extract_numbers("华东区合计 3,053,000 元") == ["3053000"]
    assert normalize_number("3,053,000") == "3053000"


def test_trivial_small_numbers_are_skipped():
    """个位数出现在任何文本里都不奇怪，追溯它们只会制造噪声。"""
    assert extract_numbers("有 3 个小组，共 27 人") == ["27"]


def test_numbers_are_deduplicated():
    assert extract_numbers("32 人，其中 32 人在职") == ["32"]


# ---------------------------------------------------------------- 抽标识符

def test_extracts_project_and_ticket_codes():
    ids = extract_identifiers("Beta 项目（PRJ-2026-0142）和 TICKET-88231 有关")
    assert "PRJ-2026-0142" in ids
    assert "TICKET-88231" in ids


def test_extracts_product_models_and_acronyms():
    ids = extract_identifiers("X7-Pro 的 SLA 是 99.95%")
    assert "X7-Pro" in ids
    assert "SLA" in ids


def test_identifiers_are_case_insensitively_deduplicated():
    assert len(extract_identifiers("SLA 和 sla-x 里的 SLA")) == 2


# ---------------------------------------------------------------- 溯源判定

EVIDENCE = [
    "# 后端组\n后端组组长是张敏。后端组现有 8 人。",
    '{"columns": ["部门", "人数"], "rows": [{"部门": "技术部", "人数": 32}]}',
]


def test_number_from_document_is_grounded():
    result = check("后端组有 8 人", EVIDENCE)
    assert result["ungrounded_numbers"] == []
    assert result["grounding_rate"] == 1.0


def test_number_from_table_result_is_grounded():
    """★ 证据必须同时算两条链路。

    实测踩过：问「Delta 项目负责人管理的小组有多少人」，答案里的数字
    来自 table_query 的返回结果，只看文档证据会误判成幻觉。
    """
    assert check("技术部有 32 人", EVIDENCE)["ungrounded_numbers"] == []


def test_fabricated_number_is_caught():
    assert check("后端组有 99 人", EVIDENCE)["ungrounded_numbers"] == ["99"]


def test_fabricated_identifier_is_caught():
    """★ 标识符是最硬的判据：它不可能是"算出来的"。"""
    result = check("详见 TICKET-99999", EVIDENCE)
    assert result["ungrounded_identifiers"] == ["TICKET-99999"]


def test_computed_number_is_reported_but_not_judged():
    """★ 未溯源 ≠ 幻觉。

    问「公司总共多少人」答 84，84 不在任何证据里——它是 32+14+18+9+11 算出来的，
    完全正确。本模块只负责**列出**未溯源的数字，判断交给调用方按题型区分。
    这条测试存在是为了防止有人把 ungrounded 直接当成 hallucination。
    """
    result = check("公司总共 84 人", EVIDENCE)
    assert result["ungrounded_numbers"] == ["84"]        # 如实列出
    assert "hallucination" not in result                 # 但不下判断


def test_answer_without_claims_returns_none_not_one():
    """★ 「答案里一个数字都没有」和「所有数字都有出处」不是一回事。

    返回 1.0 会让"什么都没说"的答案拿满分，把平均分拉高。
    """
    assert check("没有查到相关内容。", EVIDENCE)["grounding_rate"] is None


def test_partial_grounding():
    result = check("后端组有 8 人，技术部有 999 人", EVIDENCE)
    assert result["ungrounded_numbers"] == ["999"]
    assert result["grounding_rate"] == 0.5


def test_empty_evidence_grounds_nothing():
    assert check("后端组有 8 人", [])["ungrounded_numbers"] == ["8"]
