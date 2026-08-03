"""建库 + 跑 schema。幂等，可以反复执行。"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg

from .config import get_config
from .db import apply_schema, close_all


def main() -> int:
    cfg = get_config()
    parsed = urlparse(cfg.database_url)
    dbname = parsed.path.lstrip("/")
    if not dbname:
        print("DATABASE_URL 里没有数据库名", file=sys.stderr)
        return 1

    admin_url = cfg.database_url.replace(f"/{dbname}", "/postgres")
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
    print(f"已应用 {schema_path.name}：identity / mod_vector / mod_table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
