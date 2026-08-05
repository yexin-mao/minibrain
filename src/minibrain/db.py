"""连接池。边界纪律的物理落点。

每个模块一个池，每个池在建连接时就把 search_path 锁死到自己的 schema。
模块代码写不出跨 schema 的查询 —— 不是靠自觉，是靠连接本身够不着。

将来要拆成独立数据库时，改的只有这个文件里的 conninfo。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_config

_pools: dict[str, ConnectionPool] = {}


def _make_pool(name: str, schema: str, *, readonly: bool = False) -> ConnectionPool:
    cfg = get_config()

    def configure(conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            # search_path 里加上 extensions：那里只有扩展提供的类型和运算符
            # （vector、gen_random_uuid），没有任何数据表。
            # 模块之间的隔离不受影响——mod_vector 的连接仍然看不见 mod_table。
            cur.execute(f"SET search_path TO {schema}, extensions")
            if readonly:
                # 只读事务 + 语句超时。不需要建 PG 角色，也就不需要超级用户权限，
                # 但一样保证 LLM 生成的 SQL 写不了任何东西、也卡不死数据库。
                cur.execute("SET default_transaction_read_only = on")
                cur.execute(f"SET statement_timeout = {cfg.table_query_timeout_ms}")
        conn.commit()

    return ConnectionPool(
        cfg.database_url,
        min_size=1,
        max_size=4,
        kwargs={"row_factory": dict_row},
        configure=configure,
        open=True,
        name=name,
    )


def _get(name: str, schema: str, *, readonly: bool = False) -> ConnectionPool:
    if name not in _pools:
        _pools[name] = _make_pool(name, schema, readonly=readonly)
    return _pools[name]


@contextmanager
def identity_db() -> Iterator[psycopg.Cursor]:
    with _get("identity", "identity").connection() as conn, conn.cursor() as cur:
        yield cur


@contextmanager
def vector_db() -> Iterator[psycopg.Cursor]:
    with _get("vector", "mod_vector").connection() as conn, conn.cursor() as cur:
        yield cur


@contextmanager
def table_db() -> Iterator[psycopg.Cursor]:
    """表格模块的读写池：建表、写行、维护登记表。"""
    with _get("table", "mod_table").connection() as conn, conn.cursor() as cur:
        yield cur


@contextmanager
def table_readonly_db() -> Iterator[psycopg.Cursor]:
    """表格模块的只读池：专门跑 LLM 生成的 SQL。"""
    with _get("table_ro", "mod_table", readonly=True).connection() as conn, conn.cursor() as cur:
        yield cur


def close_all() -> None:
    for pool in _pools.values():
        pool.close()
    _pools.clear()


def apply_schema(schema_sql: str) -> None:
    """整份 schema.sql 跑一遍。不锁 search_path，因为它要建 schema 本身。"""
    cfg = get_config()
    with psycopg.connect(cfg.database_url, autocommit=True) as conn:
        conn.execute(schema_sql)
