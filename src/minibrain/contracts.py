"""平台与模块之间的薄契约。

只定义"谁在调用""调用哪个模块""结果长什么样"。
刻意不抽象两条链路的内部模型 —— 向量链路的 chunk 和表格链路的行，
本来就不是一回事，硬抽象只会让契约层变厚。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModuleId = Literal["vector-rag", "table-rag"]

MODULE_IDS: tuple[ModuleId, ...] = ("vector-rag", "table-rag")

Visibility = Literal["private", "public"]

DocStatus = Literal["uploaded", "processing", "ready", "failed"]


@dataclass(frozen=True)
class UserContext:
    """模块唯一认识的身份形态。

    模块永远不解析 cookie、不查 identity 表、不知道 session 的存在。
    """

    user_id: str
    username: str
    is_admin: bool


@dataclass
class Evidence:
    """一条证据。答案里每句话都要能指回其中一条。"""

    module: ModuleId
    source_name: str
    location: str          # 向量链路是文件名+段序；表格链路是表名
    snippet: str
    score: float | None = None
    # Agent 一次运行内的稳定编号，例如 E1.1（第 1 次工具调用的第 1 条证据）。
    # 模块本身不负责编号；编号由 Agent 工具适配层在证据离开模块后补上。
    evidence_id: str | None = None


@dataclass(frozen=True)
class RetrievalCandidate:
    """检索调试页里某一阶段的一条候选快照。"""

    rank: int
    node_id: str
    source_name: str
    location: str
    snippet: str
    score: float | None = None


@dataclass(frozen=True)
class RetrievalStage:
    """一次检索阶段；只在显式 explain 模式下构造。"""

    name: str
    score_kind: str
    candidates: list[RetrievalCandidate]
    latency_ms: float
    note: str = ""


@dataclass(frozen=True)
class RetrievalTrace:
    """不进入 LLM 上下文的检索执行轨迹。"""

    query: str
    retrieval_query: str
    mode: str
    fetch_k: int
    business_filters: dict[str, object]
    reranker: str | None
    mmr_lambda: float | None
    stages: list[RetrievalStage]


@dataclass(frozen=True)
class RetrievalSignals:
    """从 explain trace 提取的可校准信号，不等同于 answerability 概率。"""

    dense_top_score: float | None = None
    dense_margin: float | None = None
    keyword_top_score: float | None = None
    dense_keyword_overlap_at_5: float | None = None
    fused_top_score: float | None = None
    final_evidence_count: int = 0
    calibration_status: str = "uncalibrated"
    decision: str = "uncertain"
    decision_reason: str = "尚未应用检索置信策略"
    policy_version: str | None = None


@dataclass
class SearchResult:
    evidence: list[Evidence] = field(default_factory=list)
    note: str | None = None
    retrieval_trace: RetrievalTrace | None = None
    retrieval_signals: RetrievalSignals | None = None


class ModuleError(Exception):
    """模块对外抛出的稳定错误。不泄露内部异常细节和密钥。"""

    def __init__(self, message: str, code: str = "module_error", status: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class PermissionDenied(ModuleError):
    def __init__(self, message: str = "无权访问该资源"):
        super().__init__(message, code="permission_denied", status=403)


class NotFound(ModuleError):
    def __init__(self, message: str = "资源不存在"):
        super().__init__(message, code="not_found", status=404)
