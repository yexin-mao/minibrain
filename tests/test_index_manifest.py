from types import SimpleNamespace

from minibrain.modules.vector_rag import index_manifest
from minibrain.modules.vector_rag import chain
from minibrain.contracts import ModuleError


def test_endpoint_identity_removes_credentials_and_query():
    value = index_manifest._endpoint_identity(
        "HTTPS://alice:secret@Embed.Example.com:8443/v1/?token=hidden#fragment")

    assert value == "https://embed.example.com:8443/v1"
    assert "secret" not in value
    assert "hidden" not in value


def test_same_dimension_different_model_is_incompatible():
    current = index_manifest.IndexSignature(
        embedding_endpoint="https://embed.example/v1",
        embedding_model="model-b",
        embedding_dimensions=1024,
    )
    stored = {
        "embedding_endpoint": "https://embed.example/v1",
        "embedding_model": "model-a",
        "embedding_dimensions": 1024,
        "transform_version": index_manifest.TRANSFORM_VERSION,
    }

    differences = index_manifest.signature_differences(stored, current)

    assert differences == {
        "embedding_model": {"stored": "model-a", "configured": "model-b"},
    }


def test_transform_change_requires_reindex_even_when_model_matches():
    current = index_manifest.IndexSignature(
        embedding_endpoint="https://embed.example/v1",
        embedding_model="model-a",
        embedding_dimensions=1024,
    )
    stored = {
        **current.__dict__,
        "transform_version": "old-transform",
    }

    assert "transform_version" in index_manifest.signature_differences(stored, current)


def test_current_signature_does_not_store_api_key(monkeypatch):
    monkeypatch.setattr(index_manifest, "get_config", lambda: SimpleNamespace(
        embedding_base_url="https://user:key@example.com/v1?api_key=secret",
        embedding_model="model-a", embedding_dimensions=1024,
    ))

    signature = index_manifest.current_signature()

    assert signature.embedding_endpoint == "https://example.com/v1"


def test_chain_validates_manifest_even_when_index_object_is_cached(monkeypatch):
    old_index = chain._index
    old_validated = chain._manifest_validated
    chain._index = object()
    chain._manifest_validated = False

    def reject():
        raise ModuleError("mismatch", code="embedding_index_mismatch")

    monkeypatch.setattr(chain, "ensure_index_compatible", reject)
    try:
        try:
            chain.get_index()
        except ModuleError as exc:
            assert exc.code == "embedding_index_mismatch"
        else:
            raise AssertionError("manifest 不兼容时必须阻止读取缓存的 index 对象")
    finally:
        chain._index = old_index
        chain._manifest_validated = old_validated
