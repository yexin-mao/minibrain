"""生成人工盲标模板，或计算独立 Judge 与人工标注的一致率。

用法：
  python scripts/eval_judge_agreement.py --report result.json --init-labels labels.json
  python scripts/eval_judge_agreement.py --report result.json --labels labels.json --output agreement.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "src")

from minibrain.evaluation.agreement import (  # noqa: E402
    build_human_annotation_template,
    judge_human_agreement,
)


def _load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=pathlib.Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--init-labels", type=pathlib.Path)
    action.add_argument("--labels", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    report = _load(args.report)
    if report.get("status") != "completed":
        parser.error("只接受 status=completed 的回答质量报告")

    if args.init_labels:
        _write(args.init_labels, build_human_annotation_template(report))
        print(f"盲标模板已写入 {args.init_labels}")
        return 0

    result = judge_human_agreement(report, _load(args.labels))
    if args.output:
        _write(args.output, result)
        print(f"一致率报告已写入 {args.output}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
