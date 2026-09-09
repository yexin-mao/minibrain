"""检索缓存：TTL/LRU 之外，更重要的是权限与知识版本不能串。"""

from __future__ import annotations

from minibrain.contracts import Evidence, SearchResult
from minibrain.modules.vector_rag.cache import TTLCache, retrieval_cache_key


def test_cache_returns_copy_and_expires():
    now = [10.0]
    cache = TTLCache(max_entries=2, ttl_seconds=5, clock=lambda: now[0])
    result = SearchResult(evidence=[Evidence(
        module="vector-rag", source_name="s", location="a.md #1", snippet="正文")])
    cache.put("k", result)

    first = cache.get("k")
    assert first is not None
    first.evidence[0].evidence_id = "E1.1"
    assert cache.get("k").evidence[0].evidence_id is None

    now[0] = 15.0
    assert cache.get("k") is None


def test_cache_key_isolated_by_user_permission_and_knowledge_version():
    base = dict(query="规定是什么", top_k=5, options=("hybrid",))
    alice = retrieval_cache_key(
        user_id="alice", is_admin=False, knowledge_version="v1", **base)
    bob = retrieval_cache_key(
        user_id="bob", is_admin=False, knowledge_version="v1", **base)
    admin = retrieval_cache_key(
        user_id="alice", is_admin=True, knowledge_version="v1", **base)
    updated = retrieval_cache_key(
        user_id="alice", is_admin=False, knowledge_version="v2", **base)

    assert len({alice, bob, admin, updated}) == 4
