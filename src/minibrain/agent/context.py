"""证据上下文组装：全局去重、来源覆盖与硬 token 预算。零 LLM。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock

import tiktoken

from ..contracts import Evidence
from .tools import format_evidence


@dataclass
class ContextDecision:
    evidence_id: str | None
    module: str
    source_name: str
    location: str
    candidate_rank: int
    token_count: int
    selected: bool
    reason: str


@dataclass
class ContextMetrics:
    tokenizer: str
    token_budget: int
    evidence_limit: int
    candidate_count: int = 0
    selected_count: int = 0
    dropped_duplicate_count: int = 0
    dropped_budget_count: int = 0
    dropped_limit_count: int = 0
    dropped_per_call_limit_count: int = 0
    context_tokens: int = 0
    budget_utilization: float = 0.0


class ContextAssembler:
    """一次 Agent run 共用一个实例，因此重复检索也会跨工具调用去重。"""

    def __init__(self, *, token_budget: int, evidence_limit: int,
                 candidate_pool: int, tokenizer_name: str = "cl100k_base"):
        self.token_budget = token_budget
        self.evidence_limit = evidence_limit
        self.candidate_pool = candidate_pool
        self.tokenizer_name = tokenizer_name
        self._encoding = tiktoken.get_encoding(tokenizer_name)
        self._seen_content: set[str] = set()
        self._seen_documents: set[str] = set()
        self._selected_count = 0
        self._used_tokens = 0
        self._lock = Lock()
        self.decisions: list[ContextDecision] = []

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    @staticmethod
    def _content_key(item: Evidence) -> str:
        normalised = " ".join(item.snippet.split())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    @staticmethod
    def _document_key(item: Evidence) -> str:
        # 向量位置形如 filename #ordinal；表格以 SQL 位置作为独立证据文档。
        document = item.location.rsplit(" #", 1)[0]
        return f"{item.module}\0{item.source_name}\0{document}"

    def _tokens_for(self, item: Evidence) -> int:
        return self.count_tokens(format_evidence([item]))

    def _ordered_for_coverage(self, items: list[Evidence]) -> list[tuple[int, Evidence]]:
        """首条保持最高相关性，之后先覆盖尚未出现的文档，再补同文档片段。"""
        ranked = list(enumerate(items, start=1))
        if len(ranked) <= 1:
            return ranked
        first = ranked[0]
        seen = set(self._seen_documents)
        seen.add(self._document_key(first[1]))
        diverse: list[tuple[int, Evidence]] = []
        repeated: list[tuple[int, Evidence]] = []
        for candidate in ranked[1:]:
            key = self._document_key(candidate[1])
            if key not in seen:
                diverse.append(candidate)
                seen.add(key)
            else:
                repeated.append(candidate)
        return [first, *diverse, *repeated]

    def add(self, candidates: list[Evidence], *, max_items: int | None = None) -> list[Evidence]:
        """选择当前工具调用能进入模型上下文的证据。"""
        with self._lock:
            return self._add_locked(candidates, max_items=max_items)

    def _add_locked(self, candidates: list[Evidence], *,
                    max_items: int | None = None) -> list[Evidence]:
        selected: list[Evidence] = []
        batch_tokens = 0
        for rank, item in self._ordered_for_coverage(candidates):
            content_key = self._content_key(item)
            # BPE token 数不满足简单可加性，证据之间的分隔符也占 token。
            # 用“加入该条后整批文本 - 当前整批文本”计算边际成本，才能保证硬预算。
            prospective_tokens = self.count_tokens(format_evidence([*selected, item]))
            token_count = prospective_tokens - batch_tokens
            reason = "selected"
            keep = True
            if content_key in self._seen_content:
                keep, reason = False, "duplicate"
            elif max_items is not None and len(selected) >= max_items:
                # 给后续检索留证据槽位和 token；候选仍记进 decisions，方便诊断。
                keep, reason = False, "per_call_limit"
            elif self._selected_count >= self.evidence_limit:
                keep, reason = False, "evidence_limit"
            elif self._used_tokens + token_count > self.token_budget:
                keep, reason = False, "token_budget"

            self.decisions.append(ContextDecision(
                evidence_id=item.evidence_id,
                module=item.module,
                source_name=item.source_name,
                location=item.location,
                candidate_rank=rank,
                token_count=token_count,
                selected=keep,
                reason=reason,
            ))
            if not keep:
                continue
            self._seen_content.add(content_key)
            self._seen_documents.add(self._document_key(item))
            self._selected_count += 1
            self._used_tokens += token_count
            batch_tokens = prospective_tokens
            selected.append(item)
        return selected

    def metrics(self) -> ContextMetrics:
        with self._lock:
            return ContextMetrics(
                tokenizer=self.tokenizer_name,
                token_budget=self.token_budget,
                evidence_limit=self.evidence_limit,
                candidate_count=len(self.decisions),
                selected_count=sum(item.selected for item in self.decisions),
                dropped_duplicate_count=sum(
                    item.reason == "duplicate" for item in self.decisions),
                dropped_budget_count=sum(
                    item.reason == "token_budget" for item in self.decisions),
                dropped_limit_count=sum(
                    item.reason == "evidence_limit" for item in self.decisions),
                dropped_per_call_limit_count=sum(
                    item.reason == "per_call_limit" for item in self.decisions),
                context_tokens=self._used_tokens,
                budget_utilization=self._used_tokens / self.token_budget,
            )
