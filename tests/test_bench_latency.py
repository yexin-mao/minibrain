from __future__ import annotations

import importlib.util
import pathlib
import sys


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "bench_latency.py"
SPEC = importlib.util.spec_from_file_location("bench_latency", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_attempt_search_turns_module_error_into_measurement(monkeypatch):
    class ExpectedError(Exception):
        code = "embedding_failed"

    def fail(_user, _query):
        raise ExpectedError("network details must not abort the benchmark")

    monkeypatch.setattr(module, "_search", fail)
    total, stages, error = module._attempt_search(object(), "query")
    assert total is None
    assert stages == {}
    assert error == "embedding_failed"


def test_maybe_summarize_handles_all_requests_failing():
    assert module.maybe_summarize([]) is None
