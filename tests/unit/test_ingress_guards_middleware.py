"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


def test_before_model_normalizes_observed_malformed_tool_name():
    middleware = IngressGuardsMiddleware({"check_links"})
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_nx<-false if True else default_api:check_links",
                "args": {},
                "id": "call-1",
            }
        ],
        id="ai-1",
    )
    tool_message = ToolMessage(
        content="result",
        name="check_nx<-false if True else default_api:check_links",
        tool_call_id="call-1",
        id="tool-1",
    )

    update = middleware.before_model(
        {"messages": [ai_message, tool_message]}, runtime=SimpleNamespace()
    )

    assert update is not None
    repaired_ai = next(
        message for message in update["messages"] if message.id == "ai-1"
    )
    repaired_tool = next(
        message for message in update["messages"] if message.id == "tool-1"
    )
    assert repaired_ai.tool_calls[0]["name"] == "check_links"
    assert repaired_tool.name == "check_links"


def test_before_model_removes_unmatchable_tool_call_and_tool_message():
    middleware = IngressGuardsMiddleware({"check_links"})
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "malformed<>", "args": {}, "id": "call-1"}],
        id="ai-1",
    )
    tool_message = ToolMessage(
        content="result", name="malformed<>", tool_call_id="call-1", id="tool-1"
    )

    update = middleware.before_model(
        {"messages": [ai_message, tool_message]}, runtime=SimpleNamespace()
    )

    assert update is not None
    repaired_ai = next(
        message for message in update["messages"] if message.id == "ai-1"
    )
    assert repaired_ai.tool_calls == []
    assert any(message.id == "tool-1" for message in update["messages"])
    assert any(message.type == "remove" for message in update["messages"])


def test_before_model_keeps_well_formed_tool_name_byte_identical():
    middleware = IngressGuardsMiddleware(set())
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "check_links", "args": {}, "id": "call-1"}],
        id="ai-1",
    )

    assert (
        middleware.before_model({"messages": [ai_message]}, runtime=SimpleNamespace())
        is None
    )


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
