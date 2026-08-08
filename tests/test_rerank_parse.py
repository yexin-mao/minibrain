"""重排输出的解析。不调模型，只测 `_parse_order` 这个纯函数。

## 这个文件守的是什么

实测发现：同一个模型、同一个提示词、temperature=0，输出**极不稳定**——
有时给出 600+ token 的完整排序，有时只给 `[1]` 四个 token。

而 `_parse_order` 会把模型没提到的候选按原顺序补在后面。补齐是对的
（不补的话候选数变少，Recall 会凭空下降，分不清是重排的锅还是解析的锅），
但它带来一个后果：

    模型只给 `[1]` 时，结果 ≈ 原顺序，**等于重排根本没做事**，
    可 `fell_back` 是 False —— 代码报告「成功」。

也就是说，光看 `fell_back` **分不出「真的重排了」和「压根没排」**。
评测里混进一批空转，收益的幅度就不可信了。

所以加了 `ranked_count` / `candidate_count` 和 `partial`。
这个文件钉住那条区分：**补齐可以，但必须留痕。**
"""

from __future__ import annotations

import pytest

from minibrain.handwritten.rerank import RerankOutcome, _parse_order


def test_full_ranking_is_not_partial():
    order, reason, ranked = _parse_order("[3, 1, 2]", 3)
    assert order == [2, 0, 1]          # 提示词里是 1-based，解析成 0-based
    assert reason is None
    assert ranked == 3
    assert not RerankOutcome(order=order, fell_back=False,
                             ranked_count=ranked, candidate_count=3).partial


def test_single_element_output_is_flagged_partial():
    """★ 这就是实测抓到的那个案例：模型只回了 `[1]`。

    补齐之后 order 是 [0, 1, 2, ..., n-1] —— 和原顺序一模一样，
    重排等于没做。这时 fell_back 仍是 False，所以**必须靠 partial 才看得见**。
    """
    order, reason, ranked = _parse_order("[1]", 20)
    assert order == list(range(20)), "补齐后应当就是原顺序"
    assert reason is None                      # 解析本身是成功的
    assert ranked == 1                         # 但模型只排了 1 个
    outcome = RerankOutcome(order=order, fell_back=False,
                            ranked_count=ranked, candidate_count=20)
    assert outcome.partial, "只排了 1/20 却没被标记成 partial —— 空转会被当成成功"


def test_padding_keeps_every_candidate():
    """补齐必须补全，不能丢候选。丢了会让 Recall 凭空下降。"""
    order, _reason, ranked = _parse_order("[5, 2]", 6)
    assert sorted(order) == list(range(6))
    assert order[:2] == [4, 1]                 # 模型指定的排在最前
    assert ranked == 2


def test_out_of_range_and_duplicate_indices_are_dropped():
    """越界和重复编号要丢掉，不能污染顺序。"""
    order, _reason, ranked = _parse_order("[2, 2, 99, 0, -1, 1]", 3)
    assert ranked == 2, "只有 2 和 1 是合法的（0 和 -1 越界，99 越界，重复的 2 去重）"
    assert order[:2] == [1, 0]
    assert sorted(order) == [0, 1, 2]


@pytest.mark.parametrize("text,why", [
    ("完全没有数组的一段话", "输出里找不到 JSON 数组"),
    ("[1, 2,", "输出里找不到 JSON 数组"),      # 没闭合，正则匹配不到
    ('{"a": 1}', "输出里找不到 JSON 数组"),
    ("[]", "数组里没有任何合法编号"),
    ('["a", "b"]', "数组里没有任何合法编号"),
    ("[99, 100]", "数组里没有任何合法编号"),    # 全部越界
])
def test_unusable_output_falls_back(text, why):
    """解析不了就回落，**不猜**。

    猜出来的顺序不可信，而且会让评测测出一个「重排让指标变差」的假结论，
    真实原因却是解析出错——那种错最难查。
    """
    order, reason, ranked = _parse_order(text, 3)
    assert order is None
    assert reason == why
    assert ranked == 0


def test_json_fence_is_stripped():
    """模型爱套 ```json 围栏，得剥掉。"""
    order, reason, ranked = _parse_order("```json\n[2, 1, 3]\n```", 3)
    assert order == [1, 0, 2]
    assert reason is None
    assert ranked == 3


def test_prose_before_array_still_parses():
    """模型在数组前面写一堆推理时，仍然要能把数组抠出来。

    实测这种输出占多数（600+ token 里绝大部分是推理），
    所以这条路径必须能走通——它不是边缘情况。
    """
    text = "让我分析一下。片段 2 直接回答了问题，片段 1 提供背景。\n\n[2, 1, 3]"
    order, reason, ranked = _parse_order(text, 3)
    assert order == [1, 0, 2]
    assert reason is None
    assert ranked == 3
