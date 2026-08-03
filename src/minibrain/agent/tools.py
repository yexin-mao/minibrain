"""把两个模块的能力包成 LLM 工具。

Agent 也走 gateway，不直接 import modules —— 和 web 层同一条规矩。
工具执行永远带着当前 UserContext，模块该拦的照样拦，Agent 绕不过权限。
"""

from __future__ import annotations

import json

from .. import gateway
from ..contracts import Evidence, ModuleError, UserContext

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "在非结构化文档（报告、说明、纪要）里做语义检索，返回最相关的原文片段。"
                "适合回答'某份材料里怎么说的'这类问题。不适合做数值计算或分组统计。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索用的自然语言问题"},
                    "top_k": {"type": "integer", "description": "返回片段数，默认 5", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "table_query",
            "description": (
                "对 CSV 报表执行只读 SQL 查询，返回精确结果。"
                "凡是需要合计、平均、排序、分组、筛选的问题都必须用这个工具，不要用向量检索猜。"
                "可用的表和列已在 system prompt 中给出。标识符必须用双引号包裹。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": '单条 SELECT 语句，例：SELECT "地区", sum("销售额") FROM t_ab12cd34 GROUP BY "地区"',
                    }
                },
                "required": ["sql"],
            },
        },
    },
]


def _format_evidence(items: list[Evidence]) -> str:
    if not items:
        return "（没有命中任何内容）"
    return "\n\n".join(
        f"[{i + 1}] source={e.source_name} 位置={e.location}"
        + (f" 相似度={e.score}" if e.score is not None else "")
        + f"\n{e.snippet}"
        for i, e in enumerate(items)
    )


def execute_tool(user: UserContext, name: str, arguments: str) -> tuple[str, list[Evidence]]:
    """返回 (给 LLM 看的文本, 累积到证据面板的条目)。"""
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return "工具入参不是合法 JSON，请重试。", []

    try:
        if name == "vector_search":
            result = gateway.search(
                "vector-rag", user, args.get("query", ""), top_k=int(args.get("top_k", 5))
            )
            if result.note and not result.evidence:
                return result.note, []
            return _format_evidence(result.evidence), result.evidence

        if name == "table_query":
            payload = gateway.call("table-rag", "run_query", user, args.get("sql", ""))
            evidence = [
                Evidence(
                    module="table-rag",
                    source_name="表格查询",
                    location=payload["sql"],
                    snippet=json.dumps(payload["rows"][:20], ensure_ascii=False, default=str),
                )
            ]
            body = json.dumps(
                {"columns": payload["columns"], "rows": payload["rows"], "row_count": payload["row_count"]},
                ensure_ascii=False,
                default=str,
            )
            return f"查询成功，返回 {payload['row_count']} 行：\n{body}", evidence

        return f"未知工具 {name}", []

    except ModuleError as exc:
        # 模块的稳定错误直接回给模型，让它自己改 SQL 或换工具重试。
        return f"工具执行失败（{exc.code}）：{exc.message}", []
    except Exception as exc:
        return f"工具执行异常：{type(exc).__name__}", []
