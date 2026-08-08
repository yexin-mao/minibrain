"""RRF 融合。纯函数，期望值手算。"""

from __future__ import annotations

import pytest

from minibrain.handwritten.fusion import RRF_K, reciprocal_rank_fusion


def test_single_ranking_preserves_order():
    fused = reciprocal_rank_fusion([[2, 0, 1]])
    assert [i for i, _ in fused] == [2, 0, 1]


def test_hand_computed_scores():
    """k=60，排第 1 得 1/61，排第 2 得 1/62。"""
    fused = dict(reciprocal_rank_fusion([[5, 7]], k=60))
    assert fused[5] == pytest.approx(1 / 61)
    assert fused[7] == pytest.approx(1 / 62)


def test_agreement_between_rankings_wins():
    """两个排名都靠前的，压过只在一个排名里第一的。

    A(下标0)：两边都第 2 → 1/62 + 1/62 = 0.03226
    B(下标1)：一边第 1、一边第 3 → 1/61 + 1/63 = 0.03226 ...几乎相等
    所以用更极端的例子：C 两边都第 1，必然最高。
    """
    fused = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]])
    assert fused[0][0] == 2


def test_low_vector_rank_can_be_rescued_by_keyword():
    """★ 这就是混合检索要解决的场景。

    向量排名里正确答案（下标 7）排第 8，关键词排名里排第 1。
    融合后必须进前列——否则做这件事就没意义。
    """
    vector_ranking = [0, 1, 2, 3, 4, 5, 6, 7]     # 正确答案垫底
    keyword_ranking = [7]                          # 关键词一击命中
    fused = [i for i, _ in reciprocal_rank_fusion([vector_ranking, keyword_ranking])]
    assert fused[0] == 7


def test_document_in_only_one_ranking_still_included():
    """一路召回不到的东西，另一路仍能带进来——RRF 的重要性质。"""
    fused = [i for i, _ in reciprocal_rank_fusion([[0, 1], [9]])]
    assert 9 in fused


def test_ties_break_by_index_for_reproducibility():
    """分数完全相同时按下标升序。评测要能复现。"""
    fused = reciprocal_rank_fusion([[3, 1], [1, 3]])
    assert [i for i, _ in fused] == [1, 3]


def test_larger_k_flattens_differences():
    """k 越大，靠前名次之间的差距被压得越平。"""
    small = dict(reciprocal_rank_fusion([[0, 1]], k=1))
    large = dict(reciprocal_rank_fusion([[0, 1]], k=1000))
    assert small[0] / small[1] > large[0] / large[1]


def test_default_k_is_the_documented_value():
    assert RRF_K == 60


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# ---------------------------------------------------------------- 结构性平局
#
# 实测：项目编号 8 题里有 4 题栽在这上面。两篇文档在两个排名里
# 占据的名次集合都是 {1,2}，RRF 分数必然相同，而且跟 k 无关。

def test_symmetric_disagreement_produces_exact_tie():
    """先钉死问题本身：对称分歧下两者分数完全相等。"""
    fused = dict(reciprocal_rank_fusion([[0, 1], [1, 0]]))
    assert fused[0] == fused[1]


def test_tie_is_independent_of_k():
    """★ 调 k 解决不了——这是结构性的，不是参数问题。"""
    for k in (1, 10, 60, 1000):
        fused = dict(reciprocal_rank_fusion([[0, 1], [1, 0]], k=k))
        assert fused[0] == pytest.approx(fused[1])


def test_tie_breaker_decides_symmetric_case():
    """传入高精度那一路，平局就按它的名次破。"""
    keyword = [1, 0]
    fused = [i for i, _ in reciprocal_rank_fusion([[0, 1], keyword], tie_breaker=keyword)]
    assert fused[0] == 1


def test_tie_breaker_does_not_override_real_score_differences():
    """★ 它只管平局，不是加权——分数不等时一律以分数为准。"""
    vector = [0, 1, 2]
    keyword = [2]                       # 关键词只认第 2 篇
    fused = [i for i, _ in reciprocal_rank_fusion([vector, keyword], tie_breaker=keyword)]
    assert fused[0] == 2                # 2 拿了两路的分，确实该第一
    assert fused[1] == 0                # 0 只有向量第 1，分数高于 1，不受 tie_breaker 影响


def test_documents_absent_from_tie_breaker_sort_last_within_a_tie():
    fused = [i for i, _ in reciprocal_rank_fusion([[0, 1], [1, 0]], tie_breaker=[])]
    assert fused == [0, 1]              # 都不在 tie_breaker 里 → 退回按下标


def test_tie_breaker_trades_precision_for_coverage_knowingly():
    """★ 记录一个**已知且接受**的代价，不是 bug。

    tie_breaker 让关键词路在平局时说了算。这对编号（语料里只出现 1 次）几乎必对，
    但对 QPS / P0 这类正当出现在多篇里的缩写，BM25 会选"提得最多的"而非"下定义的"。

    实测代价：英文缩写 MRR 1.000 → 0.950（1/10 题从第 1 掉到第 2，仍在 top-5）
    实测收益：项目编号 +0.250、会议编号 +0.361、工单编号 +0.181

    这个测试存在的意义是：如果哪天有人"修好"了这个行为，
    他必须先看到这段说明，知道自己在交换什么。
    """
    vector = [0, 1]                     # 向量认为 0 更相关（定义文档）
    keyword = [1, 0]                    # 关键词认为 1 更相关（提得最多的那篇）
    fused = [i for i, _ in reciprocal_rank_fusion([vector, keyword], tie_breaker=keyword)]
    assert fused[0] == 1                # 关键词赢——这正是我们选择的行为
