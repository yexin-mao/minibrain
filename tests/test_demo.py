from minibrain import demo
from minibrain.contracts import UserContext


def test_seed_demo_registers_documents_and_skips_existing_table(monkeypatch):
    user = UserContext(user_id="u1", username="alice", is_admin=True)
    uploads: list[tuple[str, str]] = []

    def fake_call(module, method, _user, *args):
        assert _user is user
        if (module, method) == ("table-rag", "list_datasets"):
            return [{"filename": "示例-区域销售.csv"}]
        if (module, method) == ("vector-rag", "upload_document"):
            uploads.append((module, args[1]))
            return f"doc-{len(uploads)}"
        raise AssertionError((module, method))

    monkeypatch.setattr(demo.gateway, "call", fake_call)

    result = demo.seed_demo(user)

    assert result == {"documents": len(demo.DEMO_DOCUMENTS), "new_tables": 0}
    assert [filename for _, filename in uploads] == list(demo.DEMO_DOCUMENTS)


def test_seed_demo_registers_missing_table(monkeypatch):
    user = UserContext(user_id="u1", username="alice", is_admin=True)
    table_uploads: list[str] = []

    def fake_call(module, method, _user, *args):
        if (module, method) == ("table-rag", "list_datasets"):
            return []
        if (module, method) == ("vector-rag", "upload_document"):
            return "doc"
        if (module, method) == ("table-rag", "upload_bytes"):
            table_uploads.append(args[1])
            return "table"
        raise AssertionError((module, method))

    monkeypatch.setattr(demo.gateway, "call", fake_call)

    result = demo.seed_demo(user)

    assert result == {"documents": len(demo.DEMO_DOCUMENTS), "new_tables": 1}
    assert table_uploads == ["示例-区域销售.csv"]
