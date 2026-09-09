from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minibrain import gateway
from minibrain.contracts import NotFound
from minibrain.db import vector_db
from minibrain.modules.vector_rag import wiki


class _FakeCompletions:
    def __init__(self, content: str | list[str]):
        self.contents = content if isinstance(content, list) else [content]
        self.calls = 0
        self.request = None
        self.requests = []

    def create(self, **kwargs):
        self.request = kwargs
        self.requests.append(kwargs)
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeClient:
    def __init__(self, content: str | list[str]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


@pytest.fixture
def wiki_config(monkeypatch):
    cfg = SimpleNamespace(
        agent_configured=True,
        agent_base_url="https://example.invalid/v1",
        agent_api_key="test-key",
        agent_model="test-model",
        llm_timeout_seconds=1,
        llm_max_retries=0,
    )
    monkeypatch.setattr(wiki, "get_config", lambda: cfg)
    return cfg


def test_generate_summary_keeps_provenance_and_strips_fence(wiki_config):
    client = _FakeClient("```markdown\n# 差旅制度\n\n## 关键要点\n- 上限 500 元\n```")
    title, content = wiki._generate_summary(
        {"source_name": "制度库", "filename": "travel.md", "content": "住宿上限 500 元"},
        client=client,
    )

    assert title == "差旅制度"
    assert content.startswith("# 差旅制度")
    request_text = client.chat.completions.request["messages"][1]["content"]
    assert "制度库" in request_text and "travel.md" in request_text


def test_wiki_page_permissions_and_stale_lifecycle(alice, bob, wiki_config):
    source = gateway.call("vector-rag", "create_source", alice, "wiki-test")
    with vector_db() as cur:
        cur.execute(
            """INSERT INTO documents
                 (source_id, filename, content, status, char_count, content_hash, version)
               VALUES (%s, 'policy.md', '住宿上限 500 元', 'ready', 10, 'hash-v1', 1)
               RETURNING id""",
            (source["id"],),
        )
        document_id = str(cur.fetchone()["id"])

    page = wiki.build_page(
        alice,
        document_id,
        client=_FakeClient([
            "# 差旅制度\n\n- 住宿上限 500 元",
            json.dumps({"topics": [
                {"title": "住宿报销", "content": "# 住宿报销\n\n住宿上限 500 元。", "related_topics": ["交通报销"]},
                {"title": "交通报销", "content": "# 交通报销\n\n按实际票据报销。", "related_topics": ["住宿报销"]},
            ]}, ensure_ascii=False),
        ]),
    )
    assert page["status"] == "ready"
    assert {item["title"] for item in page["topics"]} == {"住宿报销", "交通报销"}
    assert len(wiki.list_pages(alice)) == 1
    assert wiki.list_pages(bob) == []
    assert len(wiki.list_topics(alice)) == 2
    assert wiki.list_topics(bob) == []
    with pytest.raises(NotFound):
        wiki.get_page(bob, str(page["id"]))

    matches = wiki.search_pages(alice, "住宿上限")
    assert {item["page_type"] for item in matches} == {"source", "topic"}
    assert wiki.search_pages(bob, "住宿上限") == []

    answer_client = _FakeClient("北京住宿上限为 500 元。[W1]")
    result = wiki.answer_question(alice, "住宿上限是多少？", client=answer_client)
    assert result["answer"].endswith("[W1]")
    assert any(ref["filename"] == "policy.md" for ref in result["references"])
    request_text = answer_client.chat.completions.request["messages"][1]["content"]
    assert "住宿上限是多少" in request_text and "住宿上限 500 元" in request_text
    assert wiki.list_events(alice)[0]["target_type"] == "query"

    topic = next(item for item in wiki.list_topics(alice) if item["title"] == "住宿报销")
    topic_detail = wiki.get_topic(alice, str(topic["id"]))
    assert topic_detail["sources"][0]["filename"] == "policy.md"
    assert {item["title"] for item in topic_detail["related_topics"]} == {"交通报销"}
    with pytest.raises(NotFound):
        wiki.get_topic(bob, str(topic["id"]))
    health = wiki.wiki_health(alice)
    assert health["topic_count"] == 2
    assert health["issue_count"] == 0
    assert len(wiki.list_events(alice)) == 4
    assert wiki.run_lint(alice)["issue_count"] == 0
    assert wiki.list_events(alice)[0]["target_type"] == "lint"

    gateway.call("vector-rag", "upload_document", alice, str(source["id"]),
                 "policy.md", "住宿上限调整为 600 元")
    revision_v1 = gateway.call(
        "vector-rag", "get_document_revision", alice, document_id, 1
    )
    revision_v2 = gateway.call(
        "vector-rag", "get_document_revision", alice, document_id, 2
    )
    assert revision_v1["content"] == "住宿上限 500 元"
    assert revision_v2["content"] == "住宿上限调整为 600 元"
    with pytest.raises(NotFound):
        gateway.call("vector-rag", "get_document_revision", bob, document_id, 1)
    assert wiki.get_page(alice, str(page["id"]))["status"] == "stale"
    assert wiki.search_pages(alice, "住宿上限") == []
    assert all(item["effective_status"] == "stale" for item in wiki.list_topics(alice))


def test_wiki_topic_compounds_multiple_sources(alice, wiki_config):
    source = gateway.call("vector-rag", "create_source", alice, "wiki-compound")
    document_ids = []
    with vector_db() as cur:
        for filename, content, digest in (
            ("travel.md", "住宿上限 500 元", "compound-a"),
            ("approval.md", "住宿超标需要负责人审批", "compound-b"),
        ):
            cur.execute(
                """INSERT INTO documents
                     (source_id, filename, content, status, char_count, content_hash, version)
                   VALUES (%s, %s, %s, 'ready', %s, %s, 1) RETURNING id""",
                (source["id"], filename, content, len(content), digest),
            )
            document_ids.append(str(cur.fetchone()["id"]))

    for index, document_id in enumerate(document_ids):
        detail = "住宿上限 500 元。" if index == 0 else "住宿上限 500 元，超标需要负责人审批。"
        wiki.build_page(
            alice,
            document_id,
            client=_FakeClient([
                f"# 来源 {index + 1}\n\n{detail}",
                json.dumps({"topics": [{
                    "title": "住宿报销",
                    "content": f"# 住宿报销\n\n{detail}",
                    "related_topics": [],
                }]}, ensure_ascii=False),
            ]),
        )

    topics = [
        item for item in wiki.list_topics(alice)
        if str(item["source_id"]) == str(source["id"])
    ]
    assert len(topics) == 1
    assert topics[0]["source_count"] == 2
    detail = wiki.get_topic(alice, str(topics[0]["id"]))
    assert {item["filename"] for item in detail["sources"]} == {"travel.md", "approval.md"}
