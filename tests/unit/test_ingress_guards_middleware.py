"""Tests for ingress message cleanup, input caps, and root-trace metadata."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
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


def test_before_agent_removes_consecutive_unanswered_human_messages():
    middleware = IngressGuardsMiddleware()
    answered = HumanMessage(content="Answered question", id="answered")
    reply = AIMessage(content="Answer", id="reply")
    orphans = [
        HumanMessage(content="Stopped question", id="orphan-1"),
        HumanMessage(content="Another stopped question", id="orphan-2"),
    ]
    current = HumanMessage(content="Current question", id="current")
    messages = [answered, reply, *orphans, current]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert all(isinstance(message, RemoveMessage) for message in update["messages"])
    assert [message.id for message in update["messages"]] == ["orphan-1", "orphan-2"]
    assert add_messages(messages, update["messages"]) == [answered, reply, current]


@pytest.mark.parametrize(
    "response",
    [
        AIMessage(content="Answer", id="reply"),
        ToolMessage(content="Tool result", tool_call_id="call-1", id="tool"),
    ],
)
def test_before_agent_preserves_human_messages_separated_by_responses(response):
    middleware = IngressGuardsMiddleware()
    messages = [
        HumanMessage(content="Previous question", id="previous"),
        response,
        HumanMessage(content="Current question", id="current"),
    ]

    assert (
        middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())
        is None
    )


@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        ("x" * (MAX_MESSAGE_CHARS + 50), "x" * MAX_MESSAGE_CHARS),
        (
            [
                "x" * (MAX_MESSAGE_CHARS - 10),
                {"type": "text", "text": "y" * 60},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image"},
                },
            ],
            [
                "x" * (MAX_MESSAGE_CHARS - 10),
                {"type": "text", "text": "y" * 10},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image"},
                },
            ],
        ),
    ],
)
def test_before_agent_merges_orphan_removal_with_input_truncation(
    content, expected_content
):
    middleware = IngressGuardsMiddleware()
    orphan = HumanMessage(content="Stopped question", id="orphan")
    current = HumanMessage(content=content, id="current")
    messages = [orphan, current]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert isinstance(update["messages"][0], RemoveMessage)
    assert update["messages"][0].id == "orphan"
    assert update["messages"][1].id == "current"
    remaining = add_messages(messages, update["messages"])
    assert len(remaining) == 1
    assert remaining[0].id == "current"
    assert remaining[0].content == expected_content


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
