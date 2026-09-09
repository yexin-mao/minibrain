"""为指定用户幂等加载内置演示知识。"""

from __future__ import annotations

import sys

from .db import close_all
from .demo import seed_demo
from .identity import list_users


def main() -> int:
    username = (sys.argv[1] if len(sys.argv) > 1 else "alice").strip()
    user = next((item for item in list_users() if item.username == username), None)
    if user is None:
        print(f"用户 {username!r} 不存在，请先创建用户。", file=sys.stderr)
        close_all()
        return 1

    try:
        result = seed_demo(user)
    finally:
        close_all()

    print(
        f"已为 {username} 登记 {result['documents']} 篇示例文档；"
        f"新增 {result['new_tables']} 张示例表。后台 worker 将继续处理索引。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
