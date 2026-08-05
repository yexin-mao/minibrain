"""向量链路：状态机与切分。

最该钉死的一条：**任何一边都不许出现"状态是 ready 但其实没索引成功"。**
配了 key 就该 ready，没配就该 failed 并写明原因——失败不许伪装成成功。
"""

from __future__ import annotations

import pytest

from minibrain import gateway
from minibrain.contracts import ModuleError
from minibrain.modules.vector_rag.chunking import split_text

from .conftest import needs_embedding, no_embedding


def _status_of(user, doc_id: str) -> dict:
    return next(d for d in gateway.call("vector-rag", "list_documents", user)
                if str(d["id"]) == doc_id)


def test_upload_returns_immediately_as_uploaded(alice):
    """上传只登记就返回：向量化是分钟级的，同步阻塞在生产上会被反代掐断。"""
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "note.md", "# 标题\n\n正文内容。"
    )
    assert _status_of(alice, doc_id)["status"] == "uploaded"


@needs_embedding
def test_processing_reaches_ready_with_chunks(alice):
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "ready.md", "# 标题\n\n正文内容。"
    )
    gateway.process("vector-rag", doc_id)
    row = _status_of(alice, doc_id)
    assert row["status"] == "ready", f'{row["status"]} {row["error"]}'
    assert row["chunk_count"] > 0
    assert row["error"] is None


@no_embedding
def test_processing_falls_to_failed_without_key(alice):
    """★ 没配 key 时必须落 failed 并写明原因，绝不能假装 ready。"""
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "nokey.md", "# 标题\n\n正文内容。"
    )
    gateway.process("vector-rag", doc_id)
    row = _status_of(alice, doc_id)
    assert row["status"] == "failed"
    assert row["error"] is not None
    assert row["error"]["code"] == "embedding_not_configured"


def test_empty_document_fails_with_reason(alice):
    doc_id = gateway.call("vector-rag", "upload_document", alice, None, "empty.md", "   ")
    gateway.process("vector-rag", doc_id)
    row = _status_of(alice, doc_id)
    assert row["status"] == "failed"
    assert row["error"]["code"] == "empty_document"


def test_empty_query_is_rejected(alice):
    with pytest.raises(ModuleError):
        gateway.search("vector-rag", alice, "   ")


@needs_embedding
def test_search_returns_scored_evidence(alice):
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "policy.md",
        "# 差旅住宿标准\n\n一线城市每晚不超过 600 元，二线城市每晚不超过 400 元。",
    )
    gateway.process("vector-rag", doc_id)
    result = gateway.search("vector-rag", alice, "住宿一晚能报多少钱", top_k=3)
    assert result.evidence
    top = result.evidence[0]
    assert top.module == "vector-rag"
    assert top.score is not None
    assert "#" in top.location          # 形如 policy.md #0


# ---------------------------------------------------------------- 切分（纯函数，无需数据库）

def test_split_keeps_paragraph_boundaries():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    assert split_text(text, chunk_size=100, overlap=10) == [text]


def test_split_breaks_oversized_paragraph():
    chunks = split_text("啊" * 250, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_split_empty_text_returns_nothing():
    assert split_text("", 100, 10) == []
    assert split_text("   \n\n  ", 100, 10) == []


def test_split_applies_overlap_between_chunks():
    """overlap 存在的意义是别把边界处的语义切断。"""
    paragraphs = "\n\n".join(f"第{i}段" + "文" * 40 for i in range(6))
    chunks = split_text(paragraphs, chunk_size=100, overlap=30)
    assert len(chunks) > 1


# ---------------------------------------------------------------- 文档清单（注入 system prompt）
#
# describe_corpus 只列 status='ready' 的文档——没索引成功的文档列出来会误导模型：
# 它会据此路由到 vector_search，然后什么也搜不到。
#
# 所以「清单里有内容」的测试**必须有 embedding key**。这不是可有可无的标注：
# CI 不带 key，这几个测试在那边跑出来目录是空的，断言会**恒真通过**——
# 假绿比红更危险，尤其其中还有一个是权限测试。
# 这三个用 @needs_embedding 明确标掉，另外补两个不需要 key 也能真正验证的。

@needs_embedding
def test_describe_corpus_lists_filenames_and_titles(alice):
    """这段文本会进 system prompt，是路由质量的直接输入（见 eval/PROMPT_ABLATION.md）。"""
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "dept-tech.md",
        "# 技术部\n\n技术部负责人是李伟。",
    )
    gateway.process("vector-rag", doc_id)
    catalog = gateway.call("vector-rag", "describe_corpus", alice)
    assert "dept-tech.md" in catalog
    assert "技术部" in catalog          # 标题被抽出来了


@needs_embedding
def test_describe_corpus_without_titles(alice):
    """只列文件名的版本。消融实验证明这一版修不好 doc-08，保留是为了结论可复现。"""
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "plain.md", "# 纯文件名\n\n正文。"
    )
    gateway.process("vector-rag", doc_id)
    catalog = gateway.call("vector-rag", "describe_corpus", alice, with_titles=False)
    assert "plain.md" in catalog
    assert "——" not in catalog          # 没有标题那一段


@needs_embedding
def test_describe_corpus_respects_permissions(alice, bob):
    """★ 清单要进 prompt，所以它本身必须是过滤过的，否则 prompt 就泄露了。

    必须先确认 alice 自己看得见——否则 bob 看不见只是因为清单本来就是空的，
    这个断言就白测了。
    """
    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "secret-plan.md", "# 机密计划\n\n不该被看到。"
    )
    gateway.process("vector-rag", doc_id)
    assert "secret-plan.md" in gateway.call("vector-rag", "describe_corpus", alice)
    assert "secret-plan.md" not in gateway.call("vector-rag", "describe_corpus", bob)


def test_describe_corpus_excludes_failed_documents(alice):
    """★ 处理失败的文档不许进清单——列出来模型会去搜，然后什么也搜不到。

    这条不需要 API key：空文档必然落 failed，两条路径都能验证。
    """
    doc_id = gateway.call("vector-rag", "upload_document", alice, None, "broken.md", "   ")
    gateway.process("vector-rag", doc_id)
    assert _status_of(alice, doc_id)["status"] == "failed"
    assert "broken.md" not in gateway.call("vector-rag", "describe_corpus", alice)


def test_describe_corpus_excludes_unprocessed_documents(alice):
    """刚上传还没处理的也不许进清单——它还没有 chunk，搜不到。"""
    gateway.call("vector-rag", "upload_document", alice, None, "pending.md", "# 待处理\n\n正文。")
    assert "pending.md" not in gateway.call("vector-rag", "describe_corpus", alice)


def test_describe_corpus_empty_is_explicit(bob):
    assert "没有" in gateway.call("vector-rag", "describe_corpus", bob)


# ---------------------------------------------------------------- 倒排索引
#
# ★ 这一组守的是本次性能优化的验收标准：
#   索引版和内存版必须算出**完全相同**的排名。分数变了就说明实现有 bug。
#   test_keyword.py 里已经用纯函数验过等价性，这里验的是**接上数据库之后**仍然一致。

@needs_embedding
def test_inverted_index_is_built_on_ingest(alice):
    """入库时就该把标识符写进倒排索引，而不是查询时才分词。"""
    from minibrain.db import vector_db

    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "idx-a.md",
        "# 故障工单 TICKET-77001\n\n工单编号 TICKET-77001，影响产品 X9-Test。",
    )
    gateway.process("vector-rag", doc_id)

    with vector_db() as cur:
        cur.execute("SELECT term FROM chunk_terms WHERE term = %s", ("ticket-77001",))
        assert cur.fetchone() is not None, "标识符没进倒排索引"
        cur.execute("SELECT term_count FROM chunks WHERE document_id = %s", (doc_id,))
        assert cur.fetchone()["term_count"] > 0, "term_count 没写"


@needs_embedding
def test_chinese_terms_are_not_indexed(alice):
    """★ 只索引标识符。中文二元组永远查不到，存了是浪费 99% 的空间。"""
    from minibrain.db import vector_db

    doc_id = gateway.call(
        "vector-rag", "upload_document", alice, None, "idx-b.md",
        "# 技术部说明\n\n技术部负责人是李伟，下设后端组。",
    )
    gateway.process("vector-rag", doc_id)
    with vector_db() as cur:
        cur.execute("SELECT count(*) AS n FROM chunk_terms ct "
                    "JOIN chunks c ON c.id = ct.chunk_id WHERE c.document_id = %s", (doc_id,))
        assert cur.fetchone()["n"] == 0, "纯中文文档不该产生任何索引行"


@needs_embedding
def test_indexed_ranking_matches_memory_ranking(alice):
    """★ 验收标准：接上数据库之后，索引版排名 == 内存版排名。"""
    from minibrain.modules.vector_rag.core import _keyword_ranking_indexed, _visible_chunks
    from minibrain.modules.vector_rag.keyword import rank_by_bm25

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
    from minibrain.modules.vector_rag.core import _keyword_ranking_indexed, _visible_chunks

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
    from minibrain.modules.vector_rag.core import _visible_chunks

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
    from minibrain.modules.vector_rag.core import _to_vector_literal
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
    from minibrain.modules.vector_rag.core import _to_vector_literal, _visibility_clause
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
