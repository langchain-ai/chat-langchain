"""Tests for ingress guards input caps, stopped requests, and trace metadata."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

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


def test_before_agent_noop_without_messages():
    middleware = IngressGuardsMiddleware()

    assert middleware.before_agent({"messages": []}, runtime=SimpleNamespace()) is None


@pytest.mark.parametrize("stopped_count", [1, 2])
@pytest.mark.parametrize("has_previous_reply", [False, True])
def test_before_agent_supersedes_stopped_requests(stopped_count, has_previous_reply):
    middleware = IngressGuardsMiddleware()
    history = [SystemMessage(content="system", id="system")]
    if has_previous_reply:
        history.extend(
            [
                HumanMessage(content="Answered question", id="answered"),
                AIMessage(content="Previous reply", id="reply"),
            ]
        )
    stopped = [
        HumanMessage(content=f"Stopped question {index}", id=f"stopped-{index}")
        for index in range(stopped_count)
    ]
    latest = HumanMessage(content="Latest question. Two sentences.", id="latest")
    messages = [*history, *stopped, latest]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    result = add_messages(messages, update["messages"])
    assert result[:-2] == history
    assert result[-1] == latest
    assert result[-1].id == "latest"
    assert isinstance(result[-2], HumanMessage)
    assert "user stopped" in result[-2].content
    assert "Answer only the latest user message" in result[-2].content
    assert "unless it explicitly refers back" in result[-2].content
    for message in stopped:
        assert f'"{message.content}"' in result[-2].content
        assert message.id not in [item.id for item in result]


@pytest.mark.parametrize(
    "previous_message",
    [
        AIMessage(content="Previous reply", id="reply"),
        ToolMessage(content="Tool result", tool_call_id="call", id="tool"),
    ],
)
def test_before_agent_preserves_history_after_ai_or_tool(previous_message):
    middleware = IngressGuardsMiddleware()
    messages = [
        HumanMessage(content="Previous question", id="previous"),
        previous_message,
        HumanMessage(content="Latest question", id="latest"),
    ]

    assert (
        middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())
        is None
    )


def test_before_agent_truncates_stopped_context_and_latest_input():
    middleware = IngressGuardsMiddleware()
    stopped = HumanMessage(content=[{"type": "text", "text": "s" * 201}], id="stopped")
    latest = HumanMessage(content="x" * (MAX_MESSAGE_CHARS + 1), id="latest")
    messages = [stopped, latest]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    result = add_messages(messages, update["messages"])
    assert f'"{"s" * 200}"' in result[-2].content
    assert "s" * 201 not in result[-2].content
    assert result[-1].id == "latest"
    assert result[-1].content == "x" * MAX_MESSAGE_CHARS


def test_before_agent_places_stopped_context_after_intervening_messages():
    middleware = IngressGuardsMiddleware()
    stopped = HumanMessage(content="Stopped question", id="stopped")
    context = SystemMessage(content="Additional context", id="context")
    latest = HumanMessage(content="Refer back to the stopped question", id="latest")
    messages = [stopped, context, latest]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    result = add_messages(messages, update["messages"])
    assert result[0] == context
    assert '"Stopped question"' in result[-2].content
    assert result[-1] == latest


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
