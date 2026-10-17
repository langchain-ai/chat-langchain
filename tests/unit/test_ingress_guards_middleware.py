"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
)
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata


def test_before_agent_truncates_oversized_human_message():
    middleware = IngressGuardsMiddleware()
    long_text = "x" * (MAX_MESSAGE_CHARS + 50)
    human = HumanMessage(content=long_text, id="h1")
    state = {"messages": [AIMessage(content="hi"), human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "h1"
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


def test_before_agent_noop_when_under_cap():
    middleware = IngressGuardsMiddleware()
    state = {"messages": [HumanMessage(content="Hello", id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_build_docs_agent_trace_metadata_includes_provenance_and_version(monkeypatch):
    monkeypatch.delenv("LANGSMITH_REVISION_ID", raising=False)
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "rev-b")
    monkeypatch.setattr(
        "src.utils.prompt_provenance._USE_LOCAL_PROMPTS",
        True,
    )

    metadata = build_docs_agent_trace_metadata()

    assert metadata["source_type"] == "Chat-LangChain"
    assert metadata["prompt_source"] == "context_hub:/instructions.md"
    assert (
        metadata["guardrails_prompt_source"]
        == "local:src/prompts/guardrails_prompts.py"
    )
    assert metadata["LANGSMITH_AGENT_VERSION"] == "rev-a"
    assert metadata["revision_id"] == "rev-a"
    assert metadata["host_revision_id"] == "rev-b"


def test_build_docs_agent_trace_metadata_falls_back_to_host_revision(monkeypatch):
    monkeypatch.delenv("LANGSMITH_REVISION_ID", raising=False)
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "host-rev")

    metadata = build_docs_agent_trace_metadata()
    assert metadata["LANGSMITH_AGENT_VERSION"] == "host-rev"
    assert metadata["host_revision_id"] == "host-rev"
    assert "revision_id" not in metadata


def test_build_docs_agent_trace_metadata_prefers_langsmith_revision(monkeypatch):
    monkeypatch.setenv("LANGSMITH_REVISION_ID", "git-sha")
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "legacy-sha")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "host-rev")

    metadata = build_docs_agent_trace_metadata()

    assert metadata["revision_id"] == "git-sha"
    assert metadata["LANGSMITH_AGENT_VERSION"] == "git-sha"
    assert metadata["host_revision_id"] == "host-rev"


def test_build_docs_agent_trace_metadata_without_revisions(monkeypatch):
    monkeypatch.delenv("LANGSMITH_REVISION_ID", raising=False)
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_HOST_REVISION_ID", raising=False)

    metadata = build_docs_agent_trace_metadata()

    assert "revision_id" not in metadata
    assert "LANGSMITH_AGENT_VERSION" not in metadata
    assert "host_revision_id" not in metadata
