from types import SimpleNamespace

from src.utils import root_run_metadata


def test_set_root_metadata_walks_to_root(monkeypatch):
    root = SimpleNamespace(extra={})
    child = SimpleNamespace(extra={}, parent_run=root)
    monkeypatch.setattr(root_run_metadata, "get_current_run_tree", lambda: child)

    root_run_metadata.set_root_metadata(guardrail_decision="blocked")

    assert root.extra == {"metadata": {"guardrail_decision": "blocked"}}
    assert child.extra == {}


def test_set_root_metadata_without_run_is_noop(monkeypatch):
    monkeypatch.setattr(root_run_metadata, "get_current_run_tree", lambda: None)

    root_run_metadata.set_root_metadata(answer_model="gpt-5.4-nano")
