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


@dataclass
class SearchResult:
    evidence: list[Evidence] = field(default_factory=list)
    note: str | None = None


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
