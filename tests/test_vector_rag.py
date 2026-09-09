"""向量链路：状态机与切分。

最该钉死的一条：**任何一边都不许出现"状态是 ready 但其实没索引成功"。**
配了 key 就该 ready，没配就该 failed 并写明原因——失败不许伪装成成功。
"""

from __future__ import annotations

import pytest

from minibrain import gateway
from minibrain.contracts import ModuleError
from minibrain.handwritten.chunking import split_text

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
