"""模块分发。

职责只有三件：注册模块、把 UserContext 透传下去、把模块错误归一化。
明确不做：替模块判断权限、跨模块融合检索结果、自动选模块。

它现在是一张进程内的 dispatch table。将来要把模块拆成独立 HTTP 服务时，
换的是这个文件里 call 的实现（改成带 x-ff-* header 的 fetch），
调用方一行都不用动 —— 这是留好的缝，不是欠的债。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .contracts import MODULE_IDS, ModuleError, ModuleId, SearchResult, UserContext
from .modules.table_rag import core as table_rag
from .modules.vector_rag import core as vector_rag


@dataclass(frozen=True)
class RegisteredModule:
    id: ModuleId
    label: str
    paradigm: str
    description: str
    core: Any


_REGISTRY: dict[str, RegisteredModule] = {
    "vector-rag": RegisteredModule(
        id="vector-rag",
        label="文档语义检索",
        paradigm="切分 + 向量召回",
        description="适合非结构化文档：报告、说明、纪要。回答「某处是怎么说的」这类问题。",
        core=vector_rag,
    ),
    "table-rag": RegisteredModule(
        id="table-rag",
        label="表格结构化查询",
        paradigm="物理表 + 只读 SQL",
        description="适合 CSV 报表：明细、台账。回答「合计多少」「按维度分组」这类需要精确计算的问题。",
        core=table_rag,
    ),
}


def list_modules() -> list[RegisteredModule]:
    return [_REGISTRY[m] for m in MODULE_IDS]


def get_module(module_id: str) -> RegisteredModule:
    module = _REGISTRY.get(module_id)
    if module is None:
        raise ModuleError(f"未知模块 {module_id}", code="unknown_module", status=404)
    return module


def call(module_id: str, method: str, user: UserContext, /, *args, **kwargs) -> Any:
    """统一调用入口。UserContext 永远是第一个业务参数，模块自己判断权限。"""
    module = get_module(module_id)
    fn: Callable | None = getattr(module.core, method, None)
    if fn is None:
        raise ModuleError(f"模块 {module_id} 不支持 {method}", code="unsupported_method", status=404)
    return fn(user, *args, **kwargs)


def search(module_id: str, user: UserContext, query: str, top_k: int = 5,
           **kwargs: Any) -> SearchResult:
    return call(module_id, "search", user, query, top_k=top_k, **kwargs)


def claim_next(module_id: str) -> str | None:
    """领取一个模块任务。任务权限已在上传登记时判过，不接收 UserContext。"""
    module = get_module(module_id)
    fn: Callable | None = getattr(module.core, "claim_next", None)
    if fn is None:
        return None
    return fn()


def process(module_id: str, entity_id: str, *, claimed: bool = False) -> None:
    """后台处理入口。不带 UserContext：权限在登记阶段已经判过，
    这里跑的是模块自己的异步任务，不代表任何用户发起新的访问。"""
    get_module(module_id).core.process(entity_id, claimed=claimed)


def renew_lease(module_id: str, entity_id: str) -> bool:
    """续租模块任务；不认识心跳的模块明确返回 False。"""
    fn: Callable | None = getattr(get_module(module_id).core, "renew_lease", None)
    return bool(fn(entity_id)) if fn is not None else False


def health() -> list[dict]:
    from .db import vector_db, table_db

    status = []
    for module in list_modules():
        try:
            db = vector_db if module.id == "vector-rag" else table_db
            with db() as cur:
                cur.execute("SELECT 1")
            status.append({"module": module.id, "status": "ok"})
        except Exception as exc:
            status.append({"module": module.id, "status": "error", "detail": type(exc).__name__})
    return status
