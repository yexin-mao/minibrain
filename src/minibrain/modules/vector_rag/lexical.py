"""主路径的持久化全局 BM25；权限和业务 metadata 都在 SQL 中过滤。"""

from __future__ import annotations

import math

from llama_index.core.schema import BaseNode, NodeWithScore
from psycopg.types.json import Jsonb

from ...contracts import UserContext
from ...db import vector_index_db
from .tokenizer import identifier_tokens, index_terms, total_term_count

K1 = 1.5
B = 0.75


def index_nodes(nodes: list[BaseNode]) -> None:
    """在向量节点写入成功后，为同一批节点建立全局倒排索引。"""
    if not nodes:
        return
    stats = []
    postings = []
    for node in nodes:
        content = node.get_content()
        metadata = dict(node.metadata)
        stats.append((
            node.node_id, metadata.get("registry_document_id"),
            str(metadata["source_name"]), str(metadata["owner_id"]),
            str(metadata["visibility"]), total_term_count(content), Jsonb(metadata),
        ))
        postings.extend(
            (node.node_id, term, freq) for term, freq in index_terms(content).items())

    with vector_index_db() as cur:
        cur.executemany(
            """INSERT INTO node_lexical_stats
                 (node_id, document_id, source_name, owner_id, visibility, term_count, metadata_)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (node_id) DO UPDATE SET
                 document_id = EXCLUDED.document_id,
                 source_name = EXCLUDED.source_name,
                 owner_id = EXCLUDED.owner_id,
                 visibility = EXCLUDED.visibility,
                 term_count = EXCLUDED.term_count,
                 metadata_ = EXCLUDED.metadata_""",
            stats,
        )
        node_ids = [node.node_id for node in nodes]
        cur.execute("DELETE FROM node_terms WHERE node_id = ANY(%s)", (node_ids,))
        if postings:
            cur.executemany(
                "INSERT INTO node_terms (node_id, term, freq) VALUES (%s, %s, %s)",
                postings,
            )


def delete_source(*, owner_id: str, source_name: str) -> None:
    with vector_index_db() as cur:
        cur.execute(
            "DELETE FROM node_lexical_stats WHERE owner_id = %s AND source_name = %s",
            (owner_id, source_name),
        )


def delete_document(*, owner_id: str, document_id: str) -> None:
    """删除一篇注册文档的 sparse 节点；owner_id 是额外的权限隔离护栏。"""
    with vector_index_db() as cur:
        cur.execute(
            "DELETE FROM node_lexical_stats WHERE owner_id = %s AND document_id = %s",
            (owner_id, document_id),
        )


def clear_all() -> None:
    with vector_index_db() as cur:
        cur.execute("TRUNCATE node_lexical_stats CASCADE")


def _where(user: UserContext, business: dict[str, object]) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if not user.is_admin:
        clauses.append("(s.visibility = 'public' OR s.owner_id = %s)")
        params.append(str(user.user_id))

    for key, raw in business.items():
        values = raw if isinstance(raw, list) else [raw]
        if key in ("document_kind", "filename"):
            clauses.append("s.metadata_ ->> %s = ANY(%s)")
            params.extend([key, [str(value) for value in values]])
        else:
            clauses.append("s.metadata_ -> %s ?| %s")
            params.extend([key, [str(value) for value in values]])
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def rank_node_ids(user: UserContext, query: str, top_k: int,
                  *, business: dict[str, object] | None = None) -> list[tuple[str, float]]:
    """从全部可见节点独立召回，不依赖向量候选池。"""
    terms = sorted(set(identifier_tokens(query)))
    if not terms or top_k <= 0:
        return []
    where, params = _where(user, business or {})
    with vector_index_db() as cur:
        cur.execute(
            f"SELECT count(*) AS n, coalesce(avg(term_count), 0) AS avg_len "
            f"FROM node_lexical_stats s WHERE {where}", params)
        stats = cur.fetchone()
        total = int(stats["n"])
        avg_len = float(stats["avg_len"] or 1.0)
        if total == 0:
            return []

        cur.execute(
            f"""SELECT t.term, t.node_id, t.freq, s.term_count
                FROM node_terms t
                JOIN node_lexical_stats s ON s.node_id = t.node_id
                WHERE {where} AND t.term = ANY(%s)""",
            params + [terms],
        )
        hits = cur.fetchall()
        if not hits:
            return []
        cur.execute(
            f"""SELECT t.term, count(*) AS df
                FROM node_terms t
                JOIN node_lexical_stats s ON s.node_id = t.node_id
                WHERE {where} AND t.term = ANY(%s)
                GROUP BY t.term""",
            params + [terms],
        )
        doc_freq = {str(row["term"]): int(row["df"]) for row in cur.fetchall()}

    scores: dict[str, float] = {}
    for row in hits:
        term = str(row["term"])
        freq = int(row["freq"])
        doc_len = int(row["term_count"])
        df = doc_freq[term]
        idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
        score = idf * (freq * (K1 + 1)) / (
            freq + K1 * (1 - B + B * doc_len / avg_len))
        node_id = str(row["node_id"])
        scores[node_id] = scores.get(node_id, 0.0) + score
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def fuse_rankings(vector_nodes: list[NodeWithScore], keyword_nodes: list[NodeWithScore],
                  top_k: int, *, rrf_k: int = 60) -> list[NodeWithScore]:
    """RRF 融合两条独立排名；完全平分时让精确关键词排名破平。"""
    by_id = {item.node.node_id: item for item in [*vector_nodes, *keyword_nodes]}
    scores: dict[str, float] = {}
    for ranking in (vector_nodes, keyword_nodes):
        for position, item in enumerate(ranking, start=1):
            node_id = item.node.node_id
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + position)
    keyword_order = {item.node.node_id: i for i, item in enumerate(keyword_nodes)}
    last = len(keyword_order) + 1
    ranked = sorted(scores, key=lambda node_id: (
        -scores[node_id], keyword_order.get(node_id, last), node_id))
    result = []
    for node_id in ranked[:top_k]:
        item = by_id[node_id]
        item.score = scores[node_id]
        result.append(item)
    return result
