"""按用户名前缀清理测试数据。

冒烟脚本会建真实用户和真实数据，跑完必须自己收拾干净 ——
一个会污染数据库的测试，跑几次之后就没人敢跑了。

用法：
  uv run minibrain-purge                    # 清 smoke_ / live_ 前缀（默认）
  uv run minibrain-purge tmp_ demo_         # 清指定前缀
  uv run minibrain-purge --list             # 只看会删什么，不动手
"""

from __future__ import annotations

import sys

from . import gateway, identity
from .contracts import MODULE_IDS, UserContext
from .db import close_all
from .observability import delete_user_runs

DEFAULT_PREFIXES = ("smoke_", "live_")


def purge_user(user: UserContext) -> int:
    """删掉这个用户在各模块里的全部 source，再删用户本身。

    顺序不能反：sources.owner_id 是跨 schema 的逻辑外键，没有数据库约束兜底，
    先删用户就会留下一堆无主数据。
    """
    removed = 0
    for module_id in MODULE_IDS:
        for source in gateway.call(module_id, "list_sources", user):
            # list_sources 也会返回 public source，只清自己名下的。
            if str(source["owner_id"]) == user.user_id:
                gateway.call(module_id, "delete_source", user, str(source["id"]))
                removed += 1
    delete_user_runs(user.user_id)
    identity.delete_user(user.user_id)
    return removed


def main() -> int:
    args = sys.argv[1:]
    dry_run = "--list" in args
    prefixes = tuple(a for a in args if not a.startswith("--")) or DEFAULT_PREFIXES

    targets = [u for u in identity.list_users() if u.username.startswith(prefixes)]
    if not targets:
        print(f"没有匹配 {'/'.join(prefixes)} 的用户，无需清理")
        return 0

    if dry_run:
        print(f"将删除 {len(targets)} 个用户（--list 模式，未实际删除）：")
        for u in targets:
            print(f"  {u.username}")
        return 0

    total = 0
    for u in targets:
        n = purge_user(u)
        total += n
        print(f"  已删 {u.username}（{n} 个 source 及其下全部数据）")

    print(f"\n共清理 {len(targets)} 个用户、{total} 个 source")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_all()
