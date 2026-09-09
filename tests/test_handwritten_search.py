"""手写版检索链路的测试。**主路径已经不走这条**，但实现保留为评测基线。

## 为什么这些测试搬到了单独文件

主路径换成 LlamaIndex 之后，`gateway.process("vector-rag", ...)` 走的是
`chain.ingest_raw`，节点存进 `mod_vector_li.data_nodes`，
**不再写 `mod_vector.chunks` 和 `chunk_terms`**。

而下面这些测试验的恰恰是手写版的内部结构：倒排索引建没建、
`_visible_chunks` 拉不拉 embedding、HNSW 索引走没走上。
它们留在 `test_vector_rag.py` 里会一直红，但那不是 bug——
**是它们在测一个已经不在主路径上的东西**。

所以搬过来，并且**改成直接调用 `handwritten.vector_search`**，
不再经过 gateway。这样两件事都成立：
- 主路径的测试测主路径
- 手写版作为评测基线，它的正确性仍然有测试守着

★ 这些断言背后都有实验数据（eval/RESULTS.md 探针十一、十二），
  不是随手写的：倒排索引快 87.6 倍、索引膨胀会静默劣化召回、
  「索引建了不等于用上了」翻过三次车。
"""

from __future__ import annotations

import pytest

from minibrain import gateway
from minibrain.handwritten import vector_search

from .conftest import needs_embedding


def _ingest(user, filename: str, text: str) -> str:
    """走**手写版**的入库：切分 + 向量化 + 写 chunks/chunk_terms。

    ★ 不能用 gateway.process —— 那条路现在是 LlamaIndex，不写这两张表。
    """
    doc_id = gateway.call("vector-rag", "upload_document", user, None, filename, text)
    vector_search.process_document(doc_id)
    return doc_id


# ---------------------------------------------------------------- 倒排索引
#
# ★ 这一组守的是本次性能优化的验收标准：
#   索引版和内存版必须算出**完全相同**的排名。分数变了就说明实现有 bug。
#   test_keyword.py 里已经用纯函数验过等价性，这里验的是**接上数据库之后**仍然一致。

@needs_embedding
def test_inverted_index_is_built_on_ingest(alice):
    """入库时就该把标识符写进倒排索引，而不是查询时才分词。"""
    from minibrain.db import vector_db

    doc_id = _ingest(alice, "idx-a.md",
        "# 故障工单 TICKET-77001\n\n工单编号 TICKET-77001，影响产品 X9-Test。")

    with vector_db() as cur:
        cur.execute("SELECT term FROM chunk_terms WHERE term = %s", ("ticket-77001",))
        assert cur.fetchone() is not None, "标识符没进倒排索引"
        cur.execute("SELECT term_count FROM chunks WHERE document_id = %s", (doc_id,))
        assert cur.fetchone()["term_count"] > 0, "term_count 没写"


@needs_embedding
def test_chinese_terms_are_not_indexed(alice):
    """★ 只索引标识符。中文二元组永远查不到，存了是浪费 99% 的空间。"""
    from minibrain.db import vector_db

    doc_id = _ingest(alice, "idx-b.md",
        "# 技术部说明\n\n技术部负责人是李伟，下设后端组。")
    with vector_db() as cur:
        cur.execute("SELECT count(*) AS n FROM chunk_terms ct "
                    "JOIN chunks c ON c.id = ct.chunk_id WHERE c.document_id = %s", (doc_id,))
        assert cur.fetchone()["n"] == 0, "纯中文文档不该产生任何索引行"


@needs_embedding
def test_indexed_ranking_matches_memory_ranking(alice):
    """★ 验收标准：接上数据库之后，索引版排名 == 内存版排名。"""
    from minibrain.handwritten.vector_search import _keyword_ranking_indexed, _visible_chunks
    from minibrain.handwritten.keyword import rank_by_bm25

    for name, text in [
        ("idx-c1.md", "# 工单 TICKET-77101\n\n编号 TICKET-77101，影响 X9-Alpha。"),
        ("idx-c2.md", "# 工单 TICKET-77102\n\n编号 TICKET-77102，影响 X9-Beta。"),
        ("idx-c3.md", "# 复盘 TICKET-77101\n\nTICKET-77101 的根因是配置未同步，涉及 X9-Alpha。"),
    ]:
        gateway.process("vector-rag", gateway.call(
            "vector-rag", "upload_document", alice, None, name, text))

    rows = _visible_chunks(alice)
    for query in ("TICKET-77101 是什么问题？", "X9-Alpha", "TICKET-77102"):
        indexed = _keyword_ranking_indexed(alice, query, rows)
        memory = rank_by_bm25(query, [r["content"] for r in rows])
        assert indexed == memory, f"「{query}」的排名对不上：索引 {indexed} vs 内存 {memory}"


@needs_embedding
def test_index_respects_permissions(alice, bob):
    """★ 索引也是数据。别人的片段绝不能通过关键词检索被捞出来。"""
    from minibrain.handwritten.vector_search import _keyword_ranking_indexed, _visible_chunks

    gateway.process("vector-rag", gateway.call(
        "vector-rag", "upload_document", alice, None, "idx-secret.md",
        "# 机密工单 TICKET-99001\n\n编号 TICKET-99001，不该被别人看到。"))

    assert _keyword_ranking_indexed(bob, "TICKET-99001", _visible_chunks(bob)) == []
    assert _keyword_ranking_indexed(alice, "TICKET-99001", _visible_chunks(alice)) != []


# ---------------------------------------------------------------- pgvector
#
# ★ 这一组守的是「向量检索搬进数据库」这次改动。
#   最大的风险不是算错，是**悄悄退化**——比如索引没被用上、或者又把全部数据搬了出来。

@needs_embedding
def test_embedding_is_stored_as_vector_type(alice):
    """列类型必须是 vector，不是 real[]。

    real[] 只能全搬进内存自己算；vector 才能用 <=> 和 HNSW 索引。
    """
    from minibrain.db import vector_db

    with vector_db() as cur:
        cur.execute("""
            SELECT format_type(a.atttypid, a.atttypmod) AS type
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'mod_vector' AND c.relname = 'chunks' AND a.attname = 'embedding'
        """)
        assert "vector" in cur.fetchone()["type"]


def test_hnsw_index_exists():
    """★ 没有这个索引，pgvector 会退化成全表扫描——而且不报错，只是悄悄变慢。"""
    from minibrain.db import vector_db

    with vector_db() as cur:
        cur.execute("""
            SELECT indexdef FROM pg_indexes
            WHERE schemaname = 'mod_vector' AND tablename = 'chunks'
              AND indexdef ILIKE '%hnsw%'
        """)
        row = cur.fetchone()
        assert row is not None, "HNSW 索引不存在"
        # 距离函数必须和检索时用的 <=> 一致，用错了索引会失效
        assert "vector_cosine_ops" in row["indexdef"]


@needs_embedding
def test_search_no_longer_pulls_embeddings(alice):
    """★ _visible_chunks 不许再拉 embedding 列。

    这次优化的全部收益就是「少搬 900×1024 个浮点数」。
    哪天有人为了图方便把 embedding 加回 SELECT，这个测试会红。
    """
    from minibrain.handwritten.vector_search import _visible_chunks

    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "pgv.md", "# 标题\n\n正文内容。")
    gateway.process("vector-rag", doc_id)

    rows = _visible_chunks(alice)
    assert rows, "没取到片段"
    assert "embedding" not in rows[0], "又把 embedding 全搬出来了"
    assert "term_count" in rows[0], "BM25 需要 term_count"


@needs_embedding
def test_vector_search_returns_cosine_score(alice):
    """数据库算出来的余弦要能传回来给用户看。"""
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "policy-pgv.md",
        "# 差旅住宿标准\n\n一线城市每晚不超过 600 元。")
    gateway.process("vector-rag", doc_id)

    result = gateway.call("vector-rag", "search", alice, "住宿能报多少", top_k=3, mode="vector")
    assert result.evidence
    assert 0.0 <= result.evidence[0].score <= 1.0


@needs_embedding
def test_vector_search_respects_permissions(alice, bob):
    """★ 检索搬进数据库之后，权限过滤必须仍在 SQL 的 WHERE 里。"""
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "pgv-secret.md",
        "# 机密\n\n绝密项目代号 Omega。")
    gateway.process("vector-rag", doc_id)

    seen = {e.location.split(" #")[0]
            for e in gateway.call("vector-rag", "search", bob, "绝密项目", top_k=10,
                                  mode="vector").evidence}
    assert "pgv-secret.md" not in seen


def _explain(cur, sql, params) -> str:
    cur.execute("EXPLAIN " + sql, params)
    return " ".join(str(next(iter(r.values()))) for r in cur.fetchall())


def _need_chunks(cur, minimum: int = 500) -> None:
    """规模不够就跳过。

    优化器在小表上不选 HNSW 是**正确行为**（小表全扫本来就快），不是 bug。
    实测 210 片段不走、900 片段走。不加这个门槛的话，测试会随着
    别的用例留下多少数据而随机红绿——那种测试比没有更糟。
    """
    cur.execute("SELECT count(*) AS n FROM chunks")
    n = cur.fetchone()["n"]
    if n < minimum:
        pytest.skip(f"只有 {n} 个片段，规模不足以让优化器选 HNSW 索引"
                    f"（实测阈值在 210~900 之间）。这是成本模型的正常行为。")


@needs_embedding
def test_hnsw_index_is_functional_on_unfiltered_search(alice):
    """索引本身是好的：无过滤的向量检索能走上 HNSW。

    这个测试守的是「索引建对了」——维度、算子类（vector_cosine_ops）、
    `<=>` 用的距离函数三者匹配。任何一个错了，计划里都不会出现 HNSW。

    ★ 它**不**代表生产检索走了 HNSW——那是下面那个测试的事。

    ⚠ 这个测试对数据量敏感，片段少时会跳过（见 _need_chunks）。
      也就是说它在 CI 里基本不跑。真正的守门人是 `scripts/probe_hnsw.py`
      里的 `check_plans()`——那里计划不对会直接中止测量。
      这里保留一份，是为了在本地跑完探针、库里有数据时能顺手验一下。
    """
    from minibrain.db import vector_db
    from minibrain.handwritten.vector_search import _to_vector_literal
    from minibrain.modules.vector_rag.embeddings import embed_query

    gateway.process("vector-rag", gateway.call(
        "vector-rag", "upload_document", alice, None, "hnsw-plan.md",
        "# 索引计划测试\n\n用来确认索引本身可用。"))

    literal = _to_vector_literal(embed_query("索引计划"))
    with vector_db() as cur:
        _need_chunks(cur)
        cur.execute("SET LOCAL enable_seqscan = off")
        plan = _explain(
            cur, "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 5",
            [literal])

    assert "chunks_embedding_hnsw_idx" in plan, f"索引本身就走不上：\n{plan[:400]}"


@needs_embedding
def test_production_search_does_not_use_hnsw_at_current_scale(alice):
    """★ 这是一个「记录现状」的测试，而且它**希望自己有一天变红**。

    现状（eval/RESULTS.md 探针十二，900 片段实测）：生产检索不加干预时，
    优化器选全表扫描，不用 HNSW。而且**强制它走 HNSW 也没有收益**——
    延迟 2.22ms vs 2.24ms，在噪声里；一致率 100%，一点召回都没丢。
    所以 `_vector_search_in_db` 里那行 `SET enable_seqscan = off` 被删掉了：
    没有实测收益的强制，就不该写进生产代码。

    为什么把现状钉成测试：

      - 上一版这里断言的是「生产检索走了 HNSW」。那个断言是**假的**，
        却一直没红——因为它要求 ≥500 片段才跑，测试库达不到，每次都跳过。
        **一个永远跳过的测试，等于把错误结论钉死在文档里。**
      - 钉住真实现状就有了触发器：等语料涨到优化器认为该用索引的规模，
        **这个测试会红**，提醒我们回去重新量一遍 ef 的工作点。

    它红的那天是好消息，按下面失败信息里的三步走。
    """
    from minibrain.db import vector_db
    from minibrain.handwritten.vector_search import _to_vector_literal, _visibility_clause
    from minibrain.modules.vector_rag.embeddings import embed_query

    gateway.process("vector-rag", gateway.call(
        "vector-rag", "upload_document", alice, None, "hnsw-filtered.md",
        "# 过滤计划测试\n\n用来记录生产查询默认走什么计划。"))

    where, params = _visibility_clause(alice)
    literal = _to_vector_literal(embed_query("过滤计划"))
    with vector_db() as cur:
        # ★ 不做任何 SET —— 测的就是「生产代码实际会走什么」。
        plan = _explain(
            cur,
            f"""SELECT c.id FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN sources s ON s.id = c.source_id
                WHERE {where} AND d.status = 'ready'
                ORDER BY c.embedding <=> %s::vector LIMIT 5""",
            params + [literal])

    assert "chunks_embedding_hnsw_idx" not in plan, (
        "好消息：优化器现在自己选 HNSW 了，说明数据量到了索引开始回本的规模。\n"
        "请做三件事，然后把这个测试改成断言「走了 HNSW」：\n"
        "  1. 重跑 scripts/probe_hnsw.py，重新量 ef 的工作点\n"
        "     （900 片段时 ef 完全不起作用，那个结论到此为止）\n"
        "  2. 更新 eval/RESULTS.md 探针十二\n"
        "  3. 复查 core.py 里「为什么不强制」那段注释\n"
        f"当前计划：\n{plan[:400]}")
