"""手写版的向量检索链路：入库（切分+向量化+倒排索引）+ 检索（向量/BM25/RRF）。

## 这个文件为什么在 handwritten/ 里

主路径已经换成 LlamaIndex（`modules/vector_rag/chain.py`）。
这里是**被换下来的那一版**，保留原因有三个，都不是情怀：

1. **评测基线。** 换框架到底换来了什么、代价是什么，只有两版跑同一批题
   才说得清。`SearchResult` 的形状两边一致，同一套脚本能直接跑。
2. **这些实现背后有实验数据。** BM25 的分词取舍、RRF 为什么不能加权求和、
   倒排索引快 87.6 倍、pgvector 的 HNSW 索引膨胀——
   eval/RESULTS.md 里十三个探针大半是围着这个文件转的。
3. **框架接不住的地方要有参照。** 比如 `BM25Retriever` 要求全部节点在内存里，
   而这一版专门建了 Postgres 倒排索引解决它。

## 和主路径的关系

主路径（`chain.py`）和这里**共用**：`tokenizer.py`（中文怎么切）、
`embeddings.py`（MRL 截断）、`core.py`（source/document 的 CRUD 和权限）。
不共用的正是被框架接管的那几层：切分、向量检索、BM25、融合。
"""

from __future__ import annotations

import json
from typing import Literal

from ..config import get_config
from ..contracts import Evidence, ModuleError, NotFound, SearchResult, UserContext
from ..db import vector_db
from ..modules.vector_rag.core import (
    MODULE_ID, _visibility_clause,
)
from ..modules.vector_rag.embeddings import embed_query, embed_texts
from .chunking import split_text
from .fusion import reciprocal_rank_fusion
from .keyword import bm25_from_postings, index_terms, total_term_count

SearchMode = Literal["hybrid", "vector", "keyword"]


def process_document(document_id: str) -> None:
    """后台线程执行。任何失败都必须落到 failed 状态，绝不伪装成 ready。"""
    cfg = get_config()

    try:
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'processing', updated_at = now() WHERE id = %s",
                (document_id,),
            )
            cur.execute("SELECT source_id, content FROM documents WHERE id = %s", (document_id,))
            row = cur.fetchone()
            if row is None:
                raise NotFound("document 不存在")
            source_id = str(row["source_id"])
            text = row["content"]

        pieces = split_text(text, cfg.chunk_size, cfg.chunk_overlap)
        if not pieces:
            raise ModuleError("文档内容为空，没有可索引的文本", code="empty_document")

        vectors = embed_texts(pieces)

        with vector_db() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            # 一并写入 term_count：BM25 长度归一化的分母，必须是**全量**分词的计数
            cur.executemany(
                """
                INSERT INTO chunks
                  (document_id, source_id, ordinal, content, embedding,
                   embedding_model, embedding_dim, term_count)
                VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s)
                RETURNING id
                """,
                [
                    # pgvector 的输入格式是 '[0.1,0.2,...]' 字符串，不是 Python list
                    (document_id, source_id, i, piece, _to_vector_literal(vec),
                     cfg.embedding_model, len(vec), total_term_count(piece))
                    for i, (piece, vec) in enumerate(zip(pieces, vectors))
                ],
            )

            # 建倒排索引。原来 BM25 每次查询都把全部文档重新分词——
            # 实测 900 片段时 201.65ms，增长 40.5 倍（见 eval/RESULTS.md 探针十）。
            # 分词是片段的固有属性，和查询无关，所以只该做一次。
            cur.execute(
                "SELECT id, ordinal FROM chunks WHERE document_id = %s ORDER BY ordinal",
                (document_id,),
            )
            postings = []
            for row in cur.fetchall():
                for term, freq in index_terms(pieces[row["ordinal"]]).items():
                    postings.append((str(row["id"]), source_id, term, freq))
            if postings:
                cur.executemany(
                    "INSERT INTO chunk_terms (chunk_id, source_id, term, freq) "
                    "VALUES (%s, %s, %s, %s)",
                    postings,
                )
            cur.execute(
                "UPDATE documents SET status = 'ready', error = NULL, updated_at = now() WHERE id = %s",
                (document_id,),
            )

    except Exception as exc:
        detail = exc.message if isinstance(exc, ModuleError) else f"{type(exc).__name__}: {exc}"
        code = exc.code if isinstance(exc, ModuleError) else "internal_error"
        with vector_db() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                (json.dumps({"code": code, "message": detail}, ensure_ascii=False), document_id),
            )


# ---------------------------------------------------------------- 检索

SearchMode = Literal["hybrid", "vector", "keyword"]

# 混合模式下，向量路取 top_k 的多少倍作为候选池。
# 太小 → 关键词认可的文档进不了候选，融合失效；太大 → 退化成"全搬"，白做 pgvector。
FUSION_POOL = 8


def _to_vector_literal(vec: list[float]) -> str:
    """Python list → pgvector 的输入格式 '[0.1,0.2,...]'。"""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


def _vector_search_in_db(user: UserContext, query: str, top_k: int,
                         *, exact: bool = False) -> list[dict]:
    """让**数据库**做向量检索，只返回 top_k 行。

    这是 pgvector 的全部意义所在。对比原来的做法：

        原来：SELECT 全部 900 行（含 1024 维向量，约 3.7MB）→ Python 算余弦 → 取前 5
              数据搬运 101ms，余弦计算 0.1ms

        现在：SELECT ... ORDER BY embedding <=> %s LIMIT 5
              数据库排完只发回 5 行

    **省的是数据搬运，不是向量运算**——实测余弦只占本地开销的 0.1%
    （eval/RESULTS.md 探针十一）。这个区别不测出来说不清。

    ★★ 收益来自 `ORDER BY ... LIMIT`，**不是来自 HNSW 索引**。
       这两件事常被混为一谈，本项目一度也混了（见探针十二）。
       实际情况是：带权限过滤的查询走的是全表扫描，HNSW 索引一次都没用上；
       但「排序和截断在数据库里完成、只发回 k 行」这个收益照样成立——
       它跟走不走索引无关。所以 pgvector 这一步**是有价值的，只是价值不在索引**。

    exact=True 时用 `SET LOCAL enable_indexscan = off` 强制走全表精确扫描，
    作为 HNSW 的**对照组**——HNSW 是近似算法，不留对照就量不出它丢了多少召回。
    """
    where, params = _visibility_clause(user)
    literal = _to_vector_literal(embed_query(query))

    with vector_db() as cur:
        if exact:
            # 强制不走索引 → 精确最近邻。只在本事务内生效
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_bitmapscan = off")
        else:
            # ef_search：HNSW 查询时的候选集大小。大 → 召回高、慢。
            cur.execute(f"SET LOCAL hnsw.ef_search = {get_config().hnsw_ef_search}")

            # ★★ 这里曾经有一行 `SET LOCAL enable_seqscan = off`，注释写着
            #    「不加这行 pgvector 白装」。**已删除**，理由是它没有收益。
            #
            #    900 片段实测（eval/RESULTS.md 探针十二，逐档 EXPLAIN 核对过计划）：
            #
            #        不干预 → 优化器选全表扫描      2.24ms
            #        强制走 HNSW                  2.22ms   ← 差值在噪声里
            #        一致率 100%，ef 从 10 调到 200 毫无区别
            #
            #    索引**能**走上（强制之后确实走了），但走上之后什么也没带来：
            #    900 片段太小，全表扫描本来就只要 2.2ms，而端到端有 92% 的时间
            #    在等 embedding 的网络往返（探针十一）。
            #
            #    **没有实测收益的强制不该写进生产代码**——它剥夺了优化器在数据
            #    变化后重新决策的机会，换来的是零。语料涨上去之后要重新量：
            #    tests/test_vector_rag.py 里那个
            #    test_production_search_does_not_use_hnsw_at_current_scale
            #    会在优化器改主意时变红，那就是重测的信号。
            #
            #    ⚠ 下面这行 hnsw.ef_search 目前**调了也没用**（见上表），
            #      保留是为了规模上去后配置就位，不是因为它现在有效果。
            #
            # ⚠⚠ 另有一个静默陷阱：**HNSW 索引会因为反复 upload/delete 而膨胀**。
            #     开发过程中它一度涨到 1125MB（表仅 18MB，死行 6 万），
            #     膨胀后优化器不选它、图结构也被撑坏导致召回下降，
            #     **两件事都不报错**。定期 `REINDEX INDEX chunks_embedding_hnsw_idx`。

        cur.execute(
            f"""
            SELECT c.id, c.content, c.ordinal, c.term_count,
                   d.filename, s.name AS source_name,
                   1 - (c.embedding <=> %s::vector) AS cosine
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN sources s ON s.id = c.source_id
            WHERE {where} AND d.status = 'ready'
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            [literal] + params + [literal, top_k],
        )
        return cur.fetchall()


def _visible_chunks(user: UserContext) -> list[dict]:
    where, params = _visibility_clause(user)
    with vector_db() as cur:
        cur.execute(
            f"""
            -- ★ 不再 SELECT embedding。
            -- 向量检索已经搬进数据库（_vector_search_in_db），这里只服务
            -- BM25 的长度归一化（term_count）和候选集统计。
            -- 少搬 900×1024 个浮点数，就是本次优化的全部收益。
            SELECT c.id, c.content, c.ordinal, c.term_count,
                   d.filename, s.name AS source_name
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN sources s ON s.id = c.source_id
            WHERE {where} AND d.status = 'ready'
            """,
            params,
        )
        return cur.fetchall()


def _keyword_ranking_indexed(user: UserContext, query: str, rows: list[dict]) -> list[int]:
    """走倒排索引的 BM25。只对命中查询词的片段打分，不碰其他片段。

    和内存版 rank_by_bm25 算出的分数**完全一致**——
    tests/test_keyword.py 里有等价性测试钉着。这是纯性能优化。
    """
    terms = sorted(set(index_terms(query)))
    if not terms:
        return []                       # 查询里没有标识符 → 关键词路不发声

    where, params = _visibility_clause(user)
    with vector_db() as cur:
        # 命中的 (词, 片段, 词频)。只拿回含查询词的行，不是全表
        cur.execute(
            f"""
            SELECT ct.term, ct.chunk_id, ct.freq
            FROM chunk_terms ct
            JOIN sources s ON s.id = ct.source_id
            WHERE {where} AND ct.term = ANY(%s)
            """,
            params + [terms],
        )
        hits = cur.fetchall()

        # 文档频率：每个词出现在几个**可见**片段里。
        # 用可见集合算 IDF 而不是全库——权限过滤之后剩什么，稀有度就按什么算。
        cur.execute(
            f"""
            SELECT ct.term, count(*) AS df
            FROM chunk_terms ct
            JOIN sources s ON s.id = ct.source_id
            WHERE {where} AND ct.term = ANY(%s)
            GROUP BY ct.term
            """,
            params + [terms],
        )
        doc_freq = {r["term"]: r["df"] for r in cur.fetchall()}

    if not hits:
        return []

    postings: dict[str, dict[str, int]] = {}
    for row in hits:
        postings.setdefault(row["term"], {})[str(row["chunk_id"])] = row["freq"]

    doc_lengths = {str(r["id"]): r["term_count"] for r in rows}
    total = len(rows)
    avg_len = (sum(doc_lengths.values()) / total) if total else 1.0

    scores = bm25_from_postings(query, postings, doc_freq, doc_lengths, total, avg_len)

    index_of = {str(r["id"]): i for i, r in enumerate(rows)}
    ranked = [(index_of[cid], sc) for cid, sc in scores.items() if cid in index_of]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in ranked]


def search(user: UserContext, query: str, top_k: int = 5,
           mode: SearchMode = "hybrid") -> SearchResult:
    """混合检索：向量召回 + BM25 关键词召回，RRF 融合。

    为什么要两条路（数据在 eval/RESULTS.md）：
      向量擅长「意思相近」——「住酒店能报多少」能找到「差旅住宿标准」，
      但对「前缀相同、只差几位数字」的编号几乎无能为力：
      8 篇会议纪要的余弦相似度全挤在 0.60~0.67，跨度仅 0.0738，MRR 只有 0.489。
      而 BM25 对 MTG-20260617-02 这种词是精确命中，一击到位。

      反过来，产品型号 / 英文缩写 / 罕见人名这三类向量是满分 1.000，
      关键词反而容易被"同一个词出现在多篇里"干扰。
      **两边强弱互补，所以融合，而不是替换。**

    mode 参数是为了**评测**存在的（scripts/probe_hybrid.py 要跑三种模式做对比），
    不是给调用方日常挑的。默认 hybrid。

    已知边界：向量和 BM25 都在内存里算，把可见片段全量拉进来。
    几万条以上要换 pgvector + PG 全文检索，见 SCALING.md。
    """
    query = query.strip()
    if not query:
        raise ModuleError("查询不能为空", code="empty_query")
    if mode not in ("hybrid", "vector", "keyword"):
        raise ModuleError(f"未知检索模式 {mode}", code="unknown_search_mode")

    rows = _visible_chunks(user)
    if not rows:
        return SearchResult(evidence=[], note="没有可检索的内容：当前用户可见范围内还没有处理完成的文档。")

    fetch_k = top_k

    # ★ 混合模式下取多少候选：
    # RRF 融合需要两条路各自的排名。向量路现在只返回 top-k（这正是 pgvector 的收益），
    # 但如果只取 5 个，关键词路排第 1 的文档可能根本不在向量的 5 个里，融合就没得融。
    # 所以向量路取 top_k * FUSION_POOL 作为候选池，融合后再截到 top_k。
    #
    # 这是 pgvector 带来的**新取舍**：原来向量排名覆盖全部片段（因为反正全搬进来了），
    # 现在只有候选池那么大。池子小 → 快但可能漏；池子大 → 接近原来的效果。
    # 实测数据见 eval/RESULTS.md 探针十二。
    pool = fetch_k * FUSION_POOL if mode == "hybrid" else fetch_k

    if mode == "keyword":
        rows = _visible_chunks(user)
        top = _keyword_ranking_indexed(user, query, rows)[:fetch_k]
        if not top:
            return SearchResult(evidence=[], note="关键词检索没有命中任何内容。")
        picked = [rows[i] for i in top]
        scores = [None] * len(picked)

    elif mode == "vector":
        picked = _vector_search_in_db(user, query, fetch_k)
        if not picked:
            return SearchResult(evidence=[], note="没有可检索的内容：当前用户可见范围内还没有处理完成的文档。")
        scores = [r["cosine"] for r in picked]

    else:                                            # hybrid
        vector_rows = _vector_search_in_db(user, query, pool)
        all_rows = _visible_chunks(user)
        if not all_rows:
            return SearchResult(evidence=[], note="没有可检索的内容：当前用户可见范围内还没有处理完成的文档。")

        # 两条路的排名都换算成「在 all_rows 里的下标」，才能喂给 RRF
        position = {str(r["id"]): i for i, r in enumerate(all_rows)}
        vector_ranking = [position[str(r["id"])] for r in vector_rows
                          if str(r["id"]) in position]
        keyword_ranking = _keyword_ranking_indexed(user, query, all_rows)

        fused = reciprocal_rank_fusion(
            [vector_ranking, keyword_ranking], tie_breaker=keyword_ranking)
        top = [i for i, _ in fused][:fetch_k]
        picked = [all_rows[i] for i in top]

        cosine_of = {str(r["id"]): r["cosine"] for r in vector_rows}
        scores = [cosine_of.get(str(r["id"])) for r in picked]

    picked, scores = picked[:top_k], scores[:top_k]

    evidence = [
        Evidence(
            module=MODULE_ID,
            source_name=row["source_name"],
            location=f"{row['filename']} #{row['ordinal']}",
            snippet=row["content"],
            # 报余弦相似度：RRF 分数（0.016 这种）对人没有意义，
            # 而余弦是"这段话和问题有多像"，看得懂。不在向量候选池里的留空。
            score=round(float(sc), 4) if sc is not None else None,
        )
        for row, sc in zip(picked, scores)
    ]
    return SearchResult(evidence=evidence)


# gateway 后台处理的统一入口名。两条链路各自的处理逻辑完全不同，
# 只在这一层对齐名字，不对齐内部模型。


process = process_document
