from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from nanobeir_data import (  # noqa: E402
    DatasetSpec, aggregate_document_ranking, build_dataset, document_filename,
)
from eval_nanobeir import _arms  # noqa: E402
from ablate_rerank_saved_candidates import _win_loss  # noqa: E402


SPEC = DatasetSpec("demo", "mteb/demo", "abc123")


def test_build_dataset_validates_and_keeps_document_level_qrels():
    dataset = build_dataset(
        SPEC,
        [{"_id": "d1", "title": "T", "text": "one"},
         {"_id": "d2", "title": "", "text": "two"}],
        [{"_id": "q1", "text": "question"}],
        [{"query-id": "q1", "corpus-id": "d1", "score": 1},
         {"query-id": "q1", "corpus-id": "d2", "score": 1}],
    )
    assert dataset.corpus[0].content == "# T\n\none"
    assert dataset.qrels["q1"] == frozenset({"d1", "d2"})


def test_missing_qrel_document_is_rejected():
    with pytest.raises(ValueError, match="不存在的 corpus"):
        build_dataset(
            SPEC, [{"_id": "d1", "text": "one"}],
            [{"_id": "q1", "text": "question"}],
            [{"query-id": "q1", "corpus-id": "missing", "score": 1}],
        )


def test_chunk_results_are_deduplicated_at_original_document_level():
    mapping = {"a.md": "doc-a", "b.md": "doc-b"}
    assert aggregate_document_ranking(
        ["a.md #0", "a.md #3", "b.md #0", "unknown.md #0"], mapping,
    ) == ["doc-a", "doc-b"]


def test_virtual_filenames_are_safe_and_collision_free():
    first = document_filename("NanoNQ", "doc/1 ?")
    second = document_filename("NanoNQ", "doc/2 ?")
    assert first.endswith(".md") and "/" not in first
    assert first != second


def test_ce_ablation_has_a_ce_only_arm_with_the_same_candidate_pool():
    arms = dict(_arms(with_ce=True))
    assert arms["hybrid"]["use_mmr"] is False
    assert arms["hybrid+ce"] == {
        "mode": "hybrid", "use_mmr": False,
        "rerank_pool": 40, "reranker": "ce",
    }
    assert "hybrid+ce+mmr" not in arms


def test_saved_candidate_win_loss_compares_the_same_cutoff():
    required = {"a", "b"}
    assert _win_loss(required, ["a", "x"], ["a", "b"], 2) == 1
    assert _win_loss(required, ["a", "b"], ["a", "x"], 2) == -1
    assert _win_loss(required, ["a", "x"], ["a", "y"], 2) == 0
