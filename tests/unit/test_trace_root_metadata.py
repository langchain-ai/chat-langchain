"""Tests for root-run outcome metadata."""

from __future__ import annotations

from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.utils import trace_root_metadata as trace_metadata


def test_build_metadata_labels_test_deployments(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ENV", "dev")
    monkeypatch.setattr(trace_metadata, "get_prompt_provenance", lambda _: {})

    metadata = trace_metadata.build_docs_agent_trace_metadata()

    assert metadata["environment"] == "test"


def test_trace_outcome_updates_root_with_served_fallback(monkeypatch):
    metadata = {}
    root = SimpleNamespace(
        parent_run=None,
        add_metadata=lambda values: metadata.update(values),
    )
    child = SimpleNamespace(parent_run=root)
    monkeypatch.setattr(trace_metadata.ls, "get_current_run_tree", lambda: child)
    middleware = trace_metadata.TraceOutcomeMiddleware(
        primary_model="google_genai:gemini-3.5-flash-lite"
    )
    request = ModelRequest(
        model=SimpleNamespace(model_name="openai:gpt-5.4-nano"),
        messages=[],
        state={},
        runtime=SimpleNamespace(),
    )
    response = ModelResponse(
        result=[AIMessage(content="answer", response_metadata={"model_name": "gpt-5.4-nano"})]
    )

    result = middleware.wrap_model_call(request, lambda _: response)

    assert result is response
    assert metadata == {
        "model_served": "gpt-5.4-nano",
        "model_fallback_used": True,
    }
