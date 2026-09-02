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


def test_before_agent_redacts_langsmith_key():
    middleware = IngressGuardsMiddleware()
    key = "lsv2_sk_" + "a" * 24
    human = HumanMessage(content=f"Use {key} here", id="h1")
    state = {"messages": [human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content == "Use lsv2_sk_<redacted> here"


def test_redaction_noop_preserves_content_identity():
    middleware = IngressGuardsMiddleware()
    content = "No credentials here"

    result = middleware._truncate_content(content)

    assert result == content
    assert result is content


def test_before_agent_redacts_list_content_blocks():
    middleware = IngressGuardsMiddleware()
    key = "lsv2_pt_" + "b" * 24
    human = HumanMessage(
        content=[{"type": "text", "text": f"Credential: {key}"}], id="h1"
    )
    state = {"messages": [human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content == [
        {"type": "text", "text": "Credential: lsv2_pt_<redacted>"}
    ]


def test_redaction_and_truncation_compose():
    middleware = IngressGuardsMiddleware()
    key = "lsv2_sk_" + "c" * 24
    content = f"{key} {'x' * MAX_MESSAGE_CHARS}"

    result = middleware._truncate_content(content)

    assert len(result) == MAX_MESSAGE_CHARS
    assert result.startswith("lsv2_sk_<redacted>")
    assert key not in result


def test_build_docs_agent_trace_metadata_includes_provenance_and_version(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "rev-b")
    monkeypatch.setattr(
        "src.utils.prompt_provenance._USE_LOCAL_PROMPTS",
        True,
    )

    metadata = build_docs_agent_trace_metadata()

    assert metadata["source_type"] == "Chat-LangChain"
    assert metadata["prompt_source"] == "local:instructions.md"
    assert (
        metadata["guardrails_prompt_source"]
        == "local:src/prompts/guardrails_prompts.py"
    )
    assert metadata["LANGSMITH_AGENT_VERSION"] == "rev-a"


def test_build_docs_agent_trace_metadata_falls_back_to_host_revision(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "host-rev")

    metadata = build_docs_agent_trace_metadata()
    assert metadata["LANGSMITH_AGENT_VERSION"] == "host-rev"
