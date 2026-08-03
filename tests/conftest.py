"""共享 fixture。

和 scripts/smoke.py 一样，测试建的是**真实用户和真实数据**——
没有 mock 数据库，因为要验证的恰恰是 SQL 里的权限过滤和只读事务，
mock 掉数据库等于把被测对象本身删了。

代价是必须自己收拾干净：session 级 fixture 结束时统一 purge。
"""

from __future__ import annotations

import uuid

import pytest

from minibrain import gateway, identity
from minibrain.config import get_config
from minibrain.db import close_all
from minibrain.scripts_purge import purge_user

SALES_CSV = """地区,产品,销售额
华东,A,1200
华东,B,800
华北,A,500
华北,B,300
"""


@pytest.fixture(scope="session")
def tag() -> str:
    """每次运行一个随机前缀，允许并发跑而不互相干扰。"""
    return uuid.uuid4().hex[:6]


@pytest.fixture(scope="session")
def users(tag: str):
    """alice / bob 是普通用户，admin 是管理员。

    权限测试至少需要三个身份：数据所有者、无关的第三方、管理员。
    """
    alice = identity.create_user(f"smoke_a_{tag}", "pw123456", is_admin=False)
    bob = identity.create_user(f"smoke_b_{tag}", "pw123456", is_admin=False)
    admin = identity.create_user(f"smoke_admin_{tag}", "pw123456", is_admin=True)

    yield {"alice": alice, "bob": bob, "admin": admin, "tag": tag}

    for user in (alice, bob, admin):
        purge_user(user)
    close_all()


@pytest.fixture(scope="session")
def alice(users):
    return users["alice"]


@pytest.fixture(scope="session")
def bob(users):
    return users["bob"]


@pytest.fixture(scope="session")
def admin(users):
    return users["admin"]


@pytest.fixture(scope="session")
def sales_table(alice) -> str:
    """alice 名下一张已就绪的物理表，返回表名。"""
    dataset_id = gateway.call(
        "table-rag", "upload_csv", alice, None, "sales.csv", SALES_CSV.encode()
    )
    gateway.process("table-rag", dataset_id)

    row = next(
        d for d in gateway.call("table-rag", "list_datasets", alice)
        if str(d["id"]) == dataset_id
    )
    assert row["status"] == "ready", f'入库失败：{row["status"]} {row["error"]}'
    return row["table_name"]


needs_embedding = pytest.mark.skipif(
    not get_config().embedding_configured,
    reason="EMBEDDING_API_KEY 未配置",
)

no_embedding = pytest.mark.skipif(
    get_config().embedding_configured,
    reason="配了 EMBEDDING_API_KEY，走 ready 分支",
)
