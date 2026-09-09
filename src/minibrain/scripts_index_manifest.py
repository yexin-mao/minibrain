"""查看或显式接管迁移前的向量索引 manifest。"""

from __future__ import annotations

import argparse
import json

from .db import close_all
from .modules.vector_rag.index_manifest import adopt_current_index, manifest_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adopt-current", action="store_true",
        help="确认现有索引由当前 embedding 配置生成；只用于一次性迁移",
    )
    args = parser.parse_args()
    try:
        if args.adopt_current:
            adopt_current_index()
        print(json.dumps(manifest_status(), ensure_ascii=False, indent=2, default=str))
    finally:
        close_all()


if __name__ == "__main__":
    main()
