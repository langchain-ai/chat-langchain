"""Tests for root-run LangSmith trace metadata."""

from __future__ import annotations

from src.utils import trace_root_metadata


def _stub_provenance(monkeypatch):
    monkeypatch.setattr(
        trace_root_metadata,
        "get_prompt_provenance",
        lambda graph_id: {"prompt_source": "local:test"},
    )


def test_environment_defaults_to_production(monkeypatch):
    _stub_provenance(monkeypatch)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    metadata = trace_root_metadata.build_docs_agent_trace_metadata()

    assert metadata["environment"] == "production"


def test_environment_honors_env_var(monkeypatch):
    _stub_provenance(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "test")

    metadata = trace_root_metadata.build_docs_agent_trace_metadata()

    assert metadata["environment"] == "test"
