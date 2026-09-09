"""向量文档的业务 metadata：确定性抽取，不调用 LLM。

这里只抽高置信度、能稳定落到 SQL 过滤条件里的字段。规则抽不出来就不加过滤；
检索层还会在业务过滤零命中时退回仅权限过滤，避免 metadata 误判变成静默漏召回。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePath

from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)

from ...contracts import UserContext


_REFERENCE_RE = re.compile(
    r"(?<![A-Z0-9])(?:PRJ-\d{4}-\d{4}|TICKET-\d{5}|MTG-\d{8}-\d{2})(?![A-Z0-9])",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_QUARTER_RE = re.compile(r"(?i)(?<![A-Z0-9])Q([1-4])(?![A-Z0-9])|第([一二三四1-4])季度")
_QUARTERS = {"一": "Q1", "二": "Q2", "三": "Q3", "四": "Q4"}

_KIND_PREFIXES = {
    "meeting": "meeting",
    "project": "project",
    "ticket": "ticket",
    "policy": "policy",
    "handbook": "policy",
    "report": "report",
    "long-report": "report",
    "dept": "department",
    "team": "team",
    "product": "product",
}
_QUERY_KIND_TERMS = {
    "meeting": ("会议", "纪要", "决议"),
    "project": ("项目",),
    "ticket": ("工单", "故障"),
    "policy": ("制度", "政策", "规定", "手册"),
    "report": ("报告", "季报"),
    "department": ("部门",),
    "team": ("小组", "团队"),
    "product": ("产品", "型号"),
}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _quarters(text: str) -> list[str]:
    found: list[str] = []
    for match in _QUARTER_RE.finditer(text):
        value = match.group(1) or match.group(2)
        found.append(_QUARTERS.get(value, f"Q{value}"))
    return _unique(found)


def _document_kind(filename: str) -> str | None:
    stem = PurePath(filename).stem.lower()
    for prefix, kind in sorted(_KIND_PREFIXES.items(), key=lambda item: -len(item[0])):
        if stem == prefix or stem.startswith(prefix + "-"):
            return kind
    return None


def extract_document_metadata(filename: str, text: str) -> dict[str, object]:
    """抽取可过滤字段；所有列表都存字符串，和 PG JSONB 操作符口径一致。"""
    haystack = f"{filename}\n{text}"
    suffix = PurePath(filename).suffix.lower().lstrip(".") or "text"
    metadata: dict[str, object] = {
        "file_type": suffix,
        "document_refs": _unique([m.group(0).upper() for m in _REFERENCE_RE.finditer(haystack)]),
        "years": _unique(_YEAR_RE.findall(haystack)),
        "quarters": _quarters(haystack),
        "content_hash": hashlib.sha256(_normalized_text(text).encode("utf-8")).hexdigest(),
    }
    kind = _document_kind(filename)
    if kind:
        metadata["document_kind"] = kind
    return metadata


def extract_query_metadata(query: str) -> dict[str, object]:
    """从查询抽高置信度过滤条件。

    查询里出现精确编号时只按编号过滤；编号本身常含年份，若再叠加文档类型等条件，
    容易把“会议编号对应项目负责人”这种跨文档问题错误收窄到单一类型。
    """
    refs = _unique([m.group(0).upper() for m in _REFERENCE_RE.finditer(query)])
    if refs:
        return {"document_refs": refs}

    kinds = [kind for kind, terms in _QUERY_KIND_TERMS.items()
             if any(term in query for term in terms)]
    result: dict[str, object] = {}
    if kinds:
        result["document_kind"] = _unique(kinds)
    years = _unique(_YEAR_RE.findall(query))
    if years:
        result["years"] = years
    quarters = _quarters(query)
    if quarters:
        result["quarters"] = quarters
    return result


def _permission_filters(user: UserContext) -> MetadataFilters | None:
    if user.is_admin:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(key="visibility", value="public", operator=FilterOperator.EQ),
            MetadataFilter(key="owner_id", value=str(user.user_id), operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )


def build_filters(user: UserContext, query: str, *, business: bool = True) -> MetadataFilters | None:
    """组合 `(权限 OR 权限) AND 业务条件`，最终由 PGVectorStore 下推到 SQL。"""
    permission = _permission_filters(user)
    extracted = extract_query_metadata(query) if business else {}
    business_groups: list[MetadataFilters | MetadataFilter] = []

    for key, raw_values in extracted.items():
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        if key == "document_kind":
            filters = [MetadataFilter(key=key, value=str(value), operator=FilterOperator.EQ)
                       for value in values]
        else:
            filters = [MetadataFilter(key=key, value=str(value), operator=FilterOperator.CONTAINS)
                       for value in values]
        business_groups.append(MetadataFilters(filters=filters, condition=FilterCondition.OR))

    parts: list[MetadataFilters | MetadataFilter] = []
    if permission is not None:
        parts.append(permission)
    parts.extend(business_groups)
    if not parts:
        return None
    if len(parts) == 1 and isinstance(parts[0], MetadataFilters):
        return parts[0]
    return MetadataFilters(filters=parts, condition=FilterCondition.AND)
