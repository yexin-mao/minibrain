"""建用户。第一个用户自动成为管理员。

用法：
  uv run minibrain-create-user                    # 交互输入
  uv run minibrain-create-user alice secret123    # 直接给
  uv run minibrain-create-user bob secret123 --admin
"""

from __future__ import annotations

import getpass
import sys

from .contracts import ModuleError
from .db import close_all
from .identity import create_user, user_count


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force_admin = "--admin" in sys.argv[1:]

    username = args[0] if args else input("用户名: ").strip()
    password = args[1] if len(args) > 1 else getpass.getpass("密码: ")

    try:
        first_user = user_count() == 0
        is_admin = force_admin or first_user
        user = create_user(username, password, is_admin=is_admin)
    except ModuleError as exc:
        print(f"创建失败：{exc.message}", file=sys.stderr)
        return 1
    finally:
        # 连接池的 worker 线程不是守护线程，CLI 退出前必须显式关掉。
        close_all()

    role = "管理员" if user.is_admin else "普通用户"
    note = "（第一个用户，自动设为管理员）" if first_user and not force_admin else ""
    print(f"已创建 {user.username}，角色：{role}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
