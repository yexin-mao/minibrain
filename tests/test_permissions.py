"""权限隔离。

规矩：可见性判定永远出现在 SQL 的 WHERE 里，不做查完再筛。
应用层后过滤是最容易长出越权 bug 的地方——漏一个分支就是数据泄露。

两条链路各自独立实现 _visibility_clause()，所以两边都要测。
"""

from __future__ import annotations

import pytest

from minibrain import gateway
from minibrain.contracts import ModuleError, PermissionDenied


# ---------------------------------------------------------------- 表格链路

def test_stranger_cannot_see_private_table(bob, sales_table):
    visible = {d["table_name"] for d in gateway.call("table-rag", "visible_tables", bob)}
    assert sales_table not in visible


def test_stranger_querying_table_by_name_is_denied(bob, sales_table):
    """★ 知道表名也没用：白名单是先用 SQL 算出来的，不是靠"猜不到表名"。"""
    with pytest.raises(PermissionDenied):
        gateway.call("table-rag", "run_query", bob, f"SELECT * FROM {sales_table}")


def test_admin_can_see_everything(admin, sales_table):
    visible = {d["table_name"] for d in gateway.call("table-rag", "visible_tables", admin)}
    assert sales_table in visible


def test_stranger_schema_description_excludes_private_table(bob, sales_table):
    """注入 Agent 的表结构说明必须也是过滤过的，否则 prompt 本身就泄露了。"""
    assert sales_table not in gateway.call("table-rag", "describe_schema", bob)


def test_stranger_cannot_delete_others_source(bob, alice, sales_table):
    source_id = str(gateway.call("table-rag", "ensure_default_source", alice)["id"])
    with pytest.raises(PermissionDenied):
        gateway.call("table-rag", "delete_source", bob, source_id)


# ---------------------------------------------------------------- 向量链路

def test_stranger_cannot_see_others_documents(alice, bob):
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "secret.md", "# 机密\n\n不该被看到。"
    )
    seen = {str(d["id"]) for d in gateway.call("vector-rag", "list_documents", bob)}
    assert doc_id not in seen


def test_stranger_cannot_delete_others_source(alice, bob):
    source_id = str(gateway.call("vector-rag", "ensure_default_source", alice)["id"])
    with pytest.raises(PermissionDenied):
        gateway.call("vector-rag", "delete_source", bob, source_id)


def test_non_admin_cannot_create_public_source(bob):
    with pytest.raises(PermissionDenied):
        gateway.call("vector-rag", "create_source", bob, f"public-{id(bob)}", "public")


def test_admin_can_create_public_source(admin):
    source = gateway.call("vector-rag", "create_source", admin, f"pub-{id(admin)}", "public")
    assert source["visibility"] == "public"
    gateway.call("vector-rag", "delete_source", admin, str(source["id"]))


# ---------------------------------------------------------------- gateway

def test_unknown_module_is_rejected(alice):
    with pytest.raises(ModuleError):
        gateway.call("no-such-module", "search", alice, "x")


def test_unsupported_method_is_rejected(alice):
    with pytest.raises(ModuleError):
        gateway.call("table-rag", "definitely_not_a_method", alice)


def test_gateway_lists_both_chains():
    assert [m.id for m in gateway.list_modules()] == ["vector-rag", "table-rag"]
