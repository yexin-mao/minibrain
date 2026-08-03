"""IR 指标的定义测试。期望值全部手算，不依赖实现。

这个文件存在的理由：指标算错了而不自知，比没有指标更糟——
它会让你信一个假数字，还拿去面试讲。
"""

from __future__ import annotations

import sys
from math import log2

import pytest

sys.path.insert(0, "scripts")

from ir_metrics import (                      # noqa: E402
    complete_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)

A, B, C, D = "a.md", "b.md", "c.md", "d.md"


# ---------------------------------------------------------------- Recall@k

def test_recall_all_relevant_retrieved():
    assert recall_at_k({A}, [A, B, C], 3) == 1.0


def test_recall_half_retrieved():
    """2 篇必需，前 2 名只命中 1 篇 → 0.5"""
    assert recall_at_k({A, B}, [A, C, B], 2) == 0.5


def test_recall_grows_with_k():
    """同样的排序，k 变大 recall 只可能升不可能降。"""
    ranking = [C, A, D, B]
    assert recall_at_k({A, B}, ranking, 2) == 0.5
    assert recall_at_k({A, B}, ranking, 4) == 1.0


def test_recall_nothing_retrieved():
    assert recall_at_k({A}, [B, C, D], 3) == 0.0


# ---------------------------------------------------------------- Precision@k

def test_precision_is_divided_by_k_not_by_result_count():
    """★ 分母是 k，不是实际返回条数——这是最常见的实现错误。"""
    assert precision_at_k({A}, [A, B, C], 3) == pytest.approx(1 / 3)


def test_precision_penalises_large_k_with_few_relevant():
    """只有 1 篇相关文档时，Precision@10 最高就是 0.1。

    这不是系统差，是指标在"相关文档远少于 k"时天然偏低。
    报这个数字必须一起说明，否则会误导。
    """
    assert precision_at_k({A}, [A] + [f"x{i}.md" for i in range(9)], 10) == pytest.approx(0.1)


def test_precision_perfect_when_all_topk_relevant():
    assert precision_at_k({A, B}, [A, B, C], 2) == 1.0


# ---------------------------------------------------------------- Hit / Complete

def test_hit_needs_only_one():
    assert hit_at_k({A, B}, [A, C, D], 3) is True


def test_hit_false_when_none():
    assert hit_at_k({A, B}, [C, D], 2) is False


def test_complete_recall_needs_all():
    """★ Hit 和 Complete Recall 的区别：前者命中 1 篇就算，后者必须全中。"""
    assert hit_at_k({A, B}, [A, C, D], 3) is True
    assert complete_recall_at_k({A, B}, [A, C, D], 3) is False
    assert complete_recall_at_k({A, B}, [A, C, B], 3) is True


# ---------------------------------------------------------------- MRR

def test_reciprocal_rank_first_position():
    assert reciprocal_rank({A}, [A, B, C]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank({A}, [B, A, C]) == 0.5


def test_reciprocal_rank_only_counts_first_hit():
    """MRR 只看第一个命中的位置，后面还有多少个相关文档它不管。"""
    assert reciprocal_rank({A, B}, [C, A, B]) == 0.5


def test_reciprocal_rank_zero_when_missing():
    assert reciprocal_rank({A}, [B, C, D]) == 0.0


# ---------------------------------------------------------------- NDCG@k

def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k({A, B}, [A, B, C], 3) == pytest.approx(1.0)


def test_ndcg_penalises_lower_position():
    """相关文档从第 1 位掉到第 2 位，Recall 不变但 NDCG 下降。"""
    assert recall_at_k({A}, [A, B, C], 3) == recall_at_k({A}, [B, A, C], 3)
    assert ndcg_at_k({A}, [B, A, C], 3) == pytest.approx(1 / log2(3))
    assert ndcg_at_k({A}, [B, A, C], 3) < ndcg_at_k({A}, [A, B, C], 3)


def test_ndcg_hand_computed():
    """手算：required={A,B}，排序 [A,C,B]，k=3

    DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5      = 1.5
    IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093  = 1.63093
    NDCG = 1.5 / 1.63093 = 0.91972
    """
    expected = (1 / log2(2) + 1 / log2(4)) / (1 / log2(2) + 1 / log2(3))
    assert ndcg_at_k({A, B}, [A, C, B], 3) == pytest.approx(expected)
    assert ndcg_at_k({A, B}, [A, C, B], 3) == pytest.approx(0.91972, abs=1e-5)


def test_ndcg_zero_when_nothing_relevant():
    assert ndcg_at_k({A}, [B, C, D], 3) == 0.0


# ---------------------------------------------------------------- 汇总

def test_summarize_averages_across_queries():
    rows = [
        ({A}, [A, B, C]),        # recall@3=1.0  RR=1.0
        ({A}, [B, C, A]),        # recall@3=1.0  RR=1/3
        ({A}, [B, C, D]),        # recall@3=0.0  RR=0
    ]
    out = summarize(rows, (3,))
    assert out["queries"] == 3
    assert out["recall@3"] == pytest.approx(2 / 3)
    assert out["hit@3"] == pytest.approx(2 / 3)
    assert out["mrr"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


def test_summarize_empty_input():
    assert summarize([], (3,)) == {}
