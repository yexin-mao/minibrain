"""召回后的确定性去重与 MMR 证据选择。零 LLM、零额外 embedding 调用。"""

from __future__ import annotations

import hashlib
import math

from llama_index.core.schema import NodeWithScore

from .tokenizer import tokenize


def _normalized_content(node: NodeWithScore) -> str:
    return " ".join(node.node.get_content().split())


def _content_key(node: NodeWithScore) -> str:
    stored = node.node.metadata.get("chunk_content_hash")
    if isinstance(stored, str) and stored:
        return stored
    return hashlib.sha256(_normalized_content(node).encode("utf-8")).hexdigest()


def deduplicate_nodes(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """按 node id 和规范化正文去重；保留排名最靠前的一条。"""
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    result: list[NodeWithScore] = []
    for item in nodes:
        node_id = item.node.node_id
        content_key = _content_key(item)
        if node_id in seen_ids or content_key in seen_content:
            continue
        seen_ids.add(node_id)
        seen_content.add(content_key)
        result.append(item)
    return result


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _lexical_similarity(left: NodeWithScore, right: NodeWithScore) -> float:
    a, b = set(tokenize(_normalized_content(left))), set(tokenize(_normalized_content(right)))
    return len(a & b) / len(a | b) if a and b else 0.0


def _redundancy(left: NodeWithScore, right: NodeWithScore) -> float:
    a, b = left.node.embedding, right.node.embedding
    similarity = _cosine(a, b) if a and b else _lexical_similarity(left, right)
    # 同一文档的相邻片段即使向量没有非常接近，也应承担一个软惩罚；不是硬上限，
    # 因此单文档问题仍可选多个片段，只是在分数接近时优先覆盖其他来源。
    if left.node.metadata.get("filename") == right.node.metadata.get("filename"):
        similarity = max(similarity, 0.85)
    return max(0.0, min(1.0, similarity))


def _relevance(nodes: list[NodeWithScore]) -> list[float]:
    # RRF、余弦和 cross-encoder 的分数量纲不同，不能直接混用绝对值。
    # 输入已经按相关性排好，因此用排名百分位统一成同一把尺子。
    denominator = max(1, len(nodes))
    return [1.0 - index / denominator for index in range(len(nodes))]


def mmr_select(nodes: list[NodeWithScore], top_k: int, *, lambda_mult: float = 0.7) -> list[NodeWithScore]:
    """选择相关且互补的一组证据；输入节点应带已存 embedding。"""
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult 必须在 0 到 1 之间")
    if top_k <= 0 or not nodes:
        return []
    if len(nodes) <= top_k:
        return list(nodes)

    relevance = _relevance(nodes)
    remaining = list(range(len(nodes)))
    selected: list[int] = []
    while remaining and len(selected) < top_k:
        def score(index: int) -> tuple[float, float, int]:
            redundancy = max((_redundancy(nodes[index], nodes[chosen]) for chosen in selected),
                             default=0.0)
            mmr = lambda_mult * relevance[index] - (1.0 - lambda_mult) * redundancy
            return mmr, relevance[index], -index

        winner = max(remaining, key=score)
        selected.append(winner)
        remaining.remove(winner)
    return [nodes[index] for index in selected]
