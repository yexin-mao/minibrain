from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "compare_embeddings.py"
SPEC = importlib.util.spec_from_file_location("compare_embeddings", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_transform_vector_truncates_and_normalizes():
    arm = module.Arm("test", "model", dimensions=2, transform="truncate_l2")
    assert module.transform_vector([3.0, 4.0, 99.0], arm) == pytest.approx([0.6, 0.8])


def test_native_dimension_mismatch_fails_loudly():
    arm = module.Arm("test", "model", dimensions=2, transform="native_l2")
    with pytest.raises(ValueError, match="期望原生 2 维"):
        module.transform_vector([1.0, 2.0, 3.0], arm)


def test_rank_documents_deduplicates_chunks_by_filename():
    chunks = [
        {"filename": "a.md", "text": "a1"},
        {"filename": "a.md", "text": "a2"},
        {"filename": "b.md", "text": "b"},
    ]
    ranking = module.rank_documents(
        [1.0, 0.0], [[1.0, 0.0], [.9, 0.0], [.8, 0.0]], chunks)
    assert ranking == ["a.md", "b.md"]
