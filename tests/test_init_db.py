"""建库脚本的连接串处理。纯函数，不碰数据库。

这个文件是 CI 第一次真跑之后补的——本地跑了几十次都没事，
CI 第一次就炸了，因为 CI 的连接串里用户名恰好等于库名。
"""

from __future__ import annotations

import pytest

from minibrain.scripts_init_db import split_database_url


def test_splits_dbname_and_admin_url():
    dbname, admin = split_database_url("postgres://myx@localhost:5432/minibrain")
    assert dbname == "minibrain"
    assert admin == "postgres://myx@localhost:5432/postgres"


def test_username_equal_to_dbname_does_not_corrupt_url():
    """★ 这就是 CI 抓到的那个 bug。

    原来的实现是 url.replace(f"/{dbname}", "/postgres")，
    str.replace 会替换**所有**匹配——用户名等于库名时，
    连 `//minibrain` 里的那处也会被换掉，连接串被改烂。
    """
    dbname, admin = split_database_url(
        "postgres://minibrain:minibrain@localhost:5432/minibrain"
    )
    assert dbname == "minibrain"
    assert admin == "postgres://minibrain:minibrain@localhost:5432/postgres"
    assert "//minibrain:minibrain@" in admin, "用户名和密码不能被改动"


def test_password_equal_to_dbname_is_also_safe():
    dbname, admin = split_database_url("postgres://u:mydb@localhost:5432/mydb")
    assert dbname == "mydb"
    assert admin == "postgres://u:mydb@localhost:5432/postgres"


def test_host_containing_dbname_is_safe():
    dbname, admin = split_database_url("postgres://u@minibrain.example.com:5432/minibrain")
    assert dbname == "minibrain"
    assert admin == "postgres://u@minibrain.example.com:5432/postgres"


def test_query_params_are_preserved():
    """sslmode 这类参数不能在换库名的时候丢掉。"""
    _, admin = split_database_url("postgres://u@h:5432/db?sslmode=require")
    assert admin == "postgres://u@h:5432/postgres?sslmode=require"


@pytest.mark.parametrize("url", [
    "postgres://u@localhost:5432/",
    "postgres://u@localhost:5432",
])
def test_missing_dbname_returns_empty(url):
    """没有库名时返回空字符串，由 main() 报错退出。"""
    dbname, _ = split_database_url(url)
    assert dbname == ""
