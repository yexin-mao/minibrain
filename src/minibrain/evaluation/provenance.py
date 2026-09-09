"""Reproducibility metadata shared by offline evaluation reports."""

from __future__ import annotations

import hashlib
import pathlib
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any


def source_fingerprint(root: pathlib.Path) -> str:
    """Hash every input that can materially change RAG evaluation behavior."""
    paths = [
        *sorted((root / "src" / "minibrain").rglob("*.py")),
        root / "schema.sql",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "scripts" / "eval_nanobeir.py",
        root / "scripts" / "ir_metrics.py",
        root / "scripts" / "nanobeir_data.py",
        root / "scripts" / "eval_answer_quality.py",
        *sorted((root / "eval").glob("probes*.json")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(root: pathlib.Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_provenance(root: pathlib.Path, *, scope: str,
                     config: dict[str, Any]) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "source_fingerprint_sha256": source_fingerprint(root),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "config": config,
    }
