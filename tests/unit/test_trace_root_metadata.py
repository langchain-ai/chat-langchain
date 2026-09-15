"""Tests for root-run outcome metadata helpers."""

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from src.utils import trace_root_metadata


def test_update_root_run_metadata_uses_runtime_root_run_id(monkeypatch):
    calls = []

    class FakeClient:
        def update_run(self, run_id, **kwargs):
            calls.append((run_id, kwargs))

    monkeypatch.setattr(trace_root_metadata.ls, "Client", FakeClient)
    runtime = SimpleNamespace(execution_info=SimpleNamespace(run_id="root-1"))

    trace_root_metadata.update_root_run_metadata(runtime, {"served_model": "primary"})

    assert calls == [("root-1", {"extra": {"metadata": {"served_model": "primary"}}})]


def test_guard_retry_count_is_bounded_and_initialized(monkeypatch):
    updates = []
    monkeypatch.setattr(
        trace_root_metadata,
        "update_root_run_metadata",
        lambda runtime, metadata: updates.append(metadata),
    )
    messages = [HumanMessage(content="Question", id="turn-1")]
    runtime = SimpleNamespace()

    assert trace_root_metadata.ensure_guard_retry_count(runtime, messages) == 0
    for _ in range(7):
        count = trace_root_metadata.increment_guard_retry_count(runtime, messages)

    assert count == 5
    assert updates[0] == {"guard_retry_count": 0}
    assert updates[-1] == {"guard_retry_count": 5}
