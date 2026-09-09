"""建库 + 跑 schema。幂等，可以反复执行。"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg

from .config import get_config
from .db import apply_schema, close_all

ADMIN_DATABASE = "postgres"


def split_database_url(url: str) -> tuple[str, str]:
    """拆出 (数据库名, 连到 postgres 库的管理连接串)。

    要建库就得先连上别的库，所以需要把连接串里的库名换成 postgres。

    ★ 这里必须用 URL 解析，不能用字符串替换。
    曾经写的是 `url.replace(f"/{dbname}", "/postgres")`，本地一直没问题，
    但 CI 第一次跑就炸了：CI 的连接串是
        postgres://minibrain:minibrain@localhost:5432/minibrain
    用户名恰好等于库名，str.replace 会把**所有**匹配都换掉，
    连 `//minibrain` 里的那处也换了，连接串直接被改烂。

    本地的连接串是 postgres://myx@localhost:5432/minibrain，
    用户名 myx ≠ 库名，所以永远碰不到这个分支。
    """
    parsed = urlparse(url)
    dbname = parsed.path.lstrip("/")
    admin_url = urlunparse(parsed._replace(path=f"/{ADMIN_DATABASE}"))
    return dbname, admin_url


def main() -> int:
    cfg = get_config()
    dbname, admin_url = split_database_url(cfg.database_url)
    if not dbname:
        print("DATABASE_URL 里没有数据库名", file=sys.stderr)
        return 1

    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if exists:
            print(f"数据库 {dbname} 已存在")
        else:
            conn.execute(f'CREATE DATABASE "{dbname}"')
            print(f"已创建数据库 {dbname}")

    schema_path = Path(__file__).resolve().parents[2] / "schema.sql"
    try:
        apply_schema(schema_path.read_text(encoding="utf-8"))
    finally:
        close_all()
    print(
        f"已应用 {schema_path.name}：identity / mod_vector / mod_vector_li / "
        "mod_table / observability"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
