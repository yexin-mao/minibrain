"""表格链路的四层护栏。

LLM 会写 SQL，所以护栏必须是纵深的，任何一层单独都不够：
  1. 表名白名单  —— 先用 SQL 算出当前用户看得见哪些表，权限的真正落点
  2. 只允许单条 SELECT / WITH
  3. 强制外层 LIMIT
  4. 只读事务 + statement_timeout —— 在连接层面，不依赖上面三层的正确性

这个文件是整个项目里最不能退化的部分：漏一条就是越权或数据被改写。
"""

from __future__ import annotations

import pytest

from minibrain import gateway
from minibrain.contracts import ModuleError


@pytest.mark.parametrize(
    "label, statement",
    [
        ("DELETE", "DELETE FROM {t}"),
        ("UPDATE", 'UPDATE {t} SET "销售额" = 0'),
        ("INSERT", "INSERT INTO {t} VALUES ('x', 'y', 1)"),
        ("DROP", "DROP TABLE {t}"),
        ("多语句", "SELECT 1; DROP TABLE {t}"),
        ("跨 schema 读身份表", "SELECT * FROM identity.users"),
        ("读系统目录", "SELECT * FROM pg_catalog.pg_tables"),
        ("未登记的表", "SELECT * FROM some_table_that_does_not_exist"),
    ],
)
def test_dangerous_statement_is_rejected(alice, sales_table, label, statement):
    with pytest.raises(ModuleError):
        gateway.call("table-rag", "run_query", alice, statement.format(t=sales_table))


def test_empty_sql_is_rejected(alice):
    with pytest.raises(ModuleError):
        gateway.call("table-rag", "run_query", alice, "   ")


def test_result_is_capped_by_outer_limit(alice, sales_table):
    """第 3 层：整条语句被包进子查询再套 LIMIT，模型写多大的查询都跑不飞。"""
    from minibrain.config import get_config

    result = gateway.call("table-rag", "run_query", alice, f"SELECT * FROM {sales_table}")
    assert result["row_count"] <= get_config().table_query_max_rows


def test_readonly_connection_blocks_writes_even_if_whitelist_passed(alice, sales_table):
    """第 4 层单独验证：绕过前三层直接用只读池写，也必须失败。

    前三层都是应用层正则，理论上存在被构造绕过的可能（README 已声明这个已知边界）。
    这一层在连接层面兜底，不依赖上面任何一层的正确性。
    """
    import psycopg

    from minibrain.db import table_readonly_db

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        with table_readonly_db() as cur:
            cur.execute(f'UPDATE {sales_table} SET "销售额" = 0')


def test_search_path_is_locked_to_module_schema(alice):
    """边界的物理落点：模块的连接够不着别的 schema。"""
    from minibrain.db import table_db

    with table_db() as cur:
        cur.execute("SHOW search_path")
        assert "mod_table" in str(cur.fetchone())
