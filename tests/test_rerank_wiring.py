"""cross-encoder 的可选接线；不加载真模型、不连接数据库。"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from minibrain.modules.vector_rag import chain, core


@pytest.mark.parametrize("fn", [core.search, chain.search], ids=["core.search", "chain.search"])
def test_search_accepts_rerank_pool(fn):
    params = inspect.signature(fn).parameters
    assert params["rerank_pool"].default == 0
    assert params["reranker"].default == "ce"
    assert params["candidate_pool"].default == 0
    assert params["use_mmr"].default is False
    assert params["within_filename"].default is None
    assert params["within_source"].default is None


def test_ablation_script_call_matches_core_signature():
    inspect.signature(core.search).bind(
        object(), "问题", top_k=5, rerank_pool=20, reranker="ce")


@pytest.mark.parametrize("top_k,pool,expected", [
    (5, 0, 5), (5, 20, 20), (5, 3, 5), (5, 5, 5),
])
def test_fetch_k_defers_truncation(top_k, pool, expected):
    assert chain._fetch_k(top_k, pool) == expected


def test_candidate_pool_makes_mmr_ablation_fair():
    assert chain._fetch_k(10, 0, 40) == 40


def test_only_cross_encoder_is_registered():
    assert set(chain.RERANKERS) == {"ce"}
    assert not hasattr(chain, "MinibrainLLMRerank")


def test_unknown_reranker_is_rejected_only_when_enabled():
    from minibrain.contracts import ModuleError

    # 关闭重排时 reranker 参数不影响主路径，不应无故报错。
    with pytest.raises(ModuleError) as exc:
        chain.search(object(), "问题", rerank_pool=20, reranker="unknown")
    assert exc.value.code == "unknown_reranker"


def test_cross_encoder_gives_actionable_error_when_extra_missing():
    from minibrain.contracts import ModuleError

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("装了 rerank-ce extra，这条降级路径测不到")

    with pytest.raises(ModuleError) as exc:
        chain.make_cross_encoder_rerank(top_n=5)
    assert exc.value.code == "rerank_ce_not_installed"
    assert "uv sync --extra rerank-ce" in exc.value.message


def _nodes(n: int):
    from llama_index.core.schema import NodeWithScore, TextNode
    return [NodeWithScore(node=TextNode(text=f"片段{i}"), score=1.0 - i * 0.1)
            for i in range(n)]


def test_cross_encoder_strips_metadata_from_model_input():
    from llama_index.core.schema import MetadataMode

    nodes = _nodes(3)
    for i, item in enumerate(nodes):
        item.node.metadata = {
            "filename": f"f{i}.md", "owner_id": "uuid-xxx",
            "visibility": "private", "ordinal": i,
        }
    assert "uuid-xxx" in nodes[0].node.get_content(metadata_mode=MetadataMode.EMBED)

    chain.strip_metadata_for_rerank(nodes)

    for item in nodes:
        content = item.node.get_content(metadata_mode=MetadataMode.EMBED)
        assert "uuid-xxx" not in content
        assert content.strip() == item.node.get_content(metadata_mode=MetadataMode.NONE).strip()


def test_cross_encoder_model_is_cached_instead_of_loaded_per_query(monkeypatch):
    created = []

    # 核心 CI 不安装 2GB 的 rerank-ce extra；这里只验证缓存接线，
    # 用占位模块越过可选依赖探测，不加载真实模型。
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.ModuleType("sentence_transformers"))

    class FakeReranker:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.top_n = kwargs["top_n"]
            self._model = object()

        def model_copy(self, *, update, deep):
            clone = object.__new__(FakeReranker)
            clone.top_n = update["top_n"]
            clone._model = self._model
            return clone

    chain._load_cross_encoder_rerank.cache_clear()
    monkeypatch.setattr(chain, "MinibrainCrossEncoderRerank", FakeReranker)
    try:
        first = chain.make_cross_encoder_rerank(40)
        second = chain.make_cross_encoder_rerank(39)
        third = chain.make_cross_encoder_rerank(10)
    finally:
        chain._load_cross_encoder_rerank.cache_clear()

    assert first is not second
    assert first._model is second._model is third._model
    assert (first.top_n, second.top_n, third.top_n) == (40, 39, 10)
    assert len(created) == 1
