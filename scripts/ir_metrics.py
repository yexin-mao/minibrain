"""标准信息检索（IR）指标。

单独一个文件、纯函数、不碰数据库——这样定义就能被单元测试钉死。
指标算错了而不自知，比没有指标更糟：它会让你信一个假数字。

术语按 IR 领域的通行定义，**不要自创名字**：
面试和文献里都用这套词，换个说法就对不上话了。

  Recall@k      前 k 个结果里，命中了多少比例的相关文档
  Precision@k   前 k 个结果里，有多少比例是相关的
  Hit Rate@k    前 k 个里**至少**命中 1 个相关文档的查询占比
  MRR           第一个相关文档排名的倒数，对所有查询取平均
  NDCG@k        考虑排序位置的加权得分（排得越靠前得分越高）

另外还有一个**非标准**的严格变体，本项目单独定义并明确标注：

  Complete Recall@k   前 k 个里命中了**全部**相关文档的查询占比

它比 Recall@k 严格：Recall@5 = 0.8 表示"5 篇必需文档召回了 4 篇"，
听起来还行，但对"公司一共几个部门"这种题，漏掉 1 篇答案就是错的。
这个项目里很多题属于这一类，所以两个都报。
"""

from __future__ import annotations

import math


def recall_at_k(required: set[str], ranking: list[str], k: int) -> float:
    """前 k 个结果里命中了多少比例的相关文档。"""
    if not required:
        return 1.0
    return len(required & set(ranking[:k])) / len(required)


def precision_at_k(required: set[str], ranking: list[str], k: int) -> float:
    """前 k 个结果里有多少比例是相关的。

    注意：相关文档只有 1 篇时，Precision@10 最高也只有 0.1。
    这不是系统差，是这个指标在"相关文档远少于 k"的场景下天然偏低——
    报它的时候必须一起说明，否则会误导。
    """
    if k <= 0:
        return 0.0
    return len(required & set(ranking[:k])) / k


def hit_at_k(required: set[str], ranking: list[str], k: int) -> bool:
    """前 k 个里是否至少命中 1 篇相关文档。"""
    return bool(required & set(ranking[:k]))


def complete_recall_at_k(required: set[str], ranking: list[str], k: int) -> bool:
    """★ 非标准的严格变体：前 k 个里是否命中了**全部**相关文档。

    多跳和全局聚合类问题必须用这个判定——漏一篇，答案就是错的。
    """
    return required <= set(ranking[:k])


def reciprocal_rank(required: set[str], ranking: list[str]) -> float:
    """第一个相关文档排名的倒数。排第 1 得 1.0，第 2 得 0.5，一个都没有得 0。

    对所有查询取平均就是 MRR。
    """
    for position, name in enumerate(ranking, start=1):
        if name in required:
            return 1.0 / position
    return 0.0


def ndcg_at_k(required: set[str], ranking: list[str], k: int) -> float:
    """考虑排序位置的加权得分，归一化到 0~1。

    相关性是二值的（是/否相关），所以 DCG 的分子恒为 1。
    IDCG 是"理想排序"（相关文档全排在最前面）的得分，用来归一化。

    和 Recall@k 的区别：Recall 只关心有没有召回，NDCG 还关心**排在第几位**。
    两篇必需文档排在 1、2 位和排在 9、10 位，Recall@10 都是 1.0，NDCG 差很多。
    """
    if not required:
        return 1.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, name in enumerate(ranking[:k], start=1)
        if name in required
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(k, len(required)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def summarize(rows: list[tuple[set[str], list[str]]], k_grid: tuple[int, ...]) -> dict:
    """一组 (相关文档集合, 召回排序) 汇总成各项指标的平均值。"""
    n = len(rows)
    if n == 0:
        return {}

    out: dict = {"queries": n, "mrr": sum(reciprocal_rank(r, g) for r, g in rows) / n}
    for k in k_grid:
        out[f"recall@{k}"] = sum(recall_at_k(r, g, k) for r, g in rows) / n
        out[f"precision@{k}"] = sum(precision_at_k(r, g, k) for r, g in rows) / n
        out[f"hit@{k}"] = sum(hit_at_k(r, g, k) for r, g in rows) / n
        out[f"complete_recall@{k}"] = sum(complete_recall_at_k(r, g, k) for r, g in rows) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, g, k) for r, g in rows) / n
    return out
