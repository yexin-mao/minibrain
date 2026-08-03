"""清理路径。

一个会污染数据库的测试，跑几次之后就没人敢跑了——所以清理本身必须被测。

这个文件刻意用自己的一次性用户，不依赖别的测试的执行顺序：
原来 smoke.py 里的第 7 组是靠"最后一个跑"来成立的，pytest 里那种隐式顺序
是脆的（-p no:randomly、-k 过滤、并行都会打乱它）。
"""

from __future__ import annotations

import uuid

from minibrain import gateway, identity
from minibrain.db import table_db
from minibrain.scripts_purge import purge_user

from .conftest import SALES_CSV


def _orphan_tables() -> set[str]:
    """物理表不在外键图里，只靠 cascade 是删不掉的——必须显式 DROP。"""
    with table_db() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'mod_table' AND tablename LIKE 't\\_%'"
        )
        physical = {r["tablename"] for r in cur.fetchall()}
        cur.execute("SELECT table_name FROM datasets")
        registered = {r["table_name"] for r in cur.fetchall()}
    return physical - registered


def test_purge_removes_user_and_drops_physical_table():
    tag = uuid.uuid4().hex[:6]
    victim = identity.create_user(f"smoke_purge_{tag}", "pw123456", is_admin=False)

    dataset_id = gateway.call(
        "table-rag", "upload_csv", victim, None, "sales.csv", SALES_CSV.encode()
    )
    gateway.process("table-rag", dataset_id)
    table_name = next(
        d["table_name"] for d in gateway.call("table-rag", "list_datasets", victim)
        if str(d["id"]) == dataset_id
    )

    # 也放一份文档，确认两条链路都被清
    gateway.call("vector-rag", "upload_document", victim, None, "note.md", "# 标题\n\n正文。")

    before = _orphan_tables()
    purge_user(victim)

    assert victim.username not in {u.username for u in identity.list_users()}
    assert _orphan_tables() == before, f"purge 后留下了孤儿物理表：{table_name}"

    with table_db() as cur:
        cur.execute(
            "SELECT 1 FROM pg_tables WHERE schemaname = 'mod_table' AND tablename = %s",
            (table_name,),
        )
        assert cur.fetchone() is None, "物理表没有被 DROP"


def test_purge_leaves_no_orphan_tables_overall():
    """全局不变量：任何时刻，mod_table 里的物理表都该在登记表里有对应记录。"""
    assert _orphan_tables() == set()
