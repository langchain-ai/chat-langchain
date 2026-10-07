"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    STOPPED_MESSAGE_PREFIX,
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


@pytest.mark.parametrize(
    "content",
    [
        "Earlier unanswered question",
        ["Earlier unanswered question"],
        [{"type": "text", "text": "Earlier unanswered question"}],
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            "Earlier unanswered question",
            {"type": "text", "text": "More context"},
        ],
        [],
    ],
)
def test_before_agent_marks_only_trailing_earlier_humans(content):
    middleware = IngressGuardsMiddleware()
    earlier = HumanMessage(content="Answered question", id="h1")
    answer = AIMessage(content="Answer", id="a1")
    orphan = HumanMessage(content=content, id="h2")
    current = HumanMessage(content="Current question", id="h3")
    messages = [earlier, answer, orphan, current]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert len(update["messages"]) == 1
    marked = update["messages"][0]
    assert marked.id == orphan.id
    if isinstance(content, str):
        assert marked.content == STOPPED_MESSAGE_PREFIX + content
    else:
        assert marked.content == [
            {"type": "text", "text": STOPPED_MESSAGE_PREFIX},
            *content,
        ]
    assert orphan.content == content
    reconciled = add_messages(messages, update["messages"])
    assert len(reconciled) == len(messages)
    assert reconciled[0] == earlier
    assert reconciled[1] == answer
    assert reconciled[2] == marked
    assert reconciled[3] == current
    assert (
        middleware.before_agent({"messages": reconciled}, runtime=SimpleNamespace())
        is None
    )


def test_before_agent_marks_multiple_orphans_and_caps_current_question():
    middleware = IngressGuardsMiddleware()
    messages = [
        HumanMessage(content="First orphan", id="h1"),
        HumanMessage(content="Second orphan", id="h2"),
        HumanMessage(content="x" * (MAX_MESSAGE_CHARS + 1), id="h3"),
    ]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    reconciled = add_messages(messages, update["messages"])
    assert reconciled[0].content == STOPPED_MESSAGE_PREFIX + "First orphan"
    assert reconciled[1].content == STOPPED_MESSAGE_PREFIX + "Second orphan"
    assert reconciled[2].content == "x" * MAX_MESSAGE_CHARS
    assert (
        middleware.before_agent({"messages": reconciled}, runtime=SimpleNamespace())
        is None
    )


def test_before_agent_leaves_answered_humans_unmarked():
    middleware = IngressGuardsMiddleware()
    messages = [
        HumanMessage(content="First question", id="h1"),
        HumanMessage(content="Second question", id="h2"),
        AIMessage(content="Answer", id="a1"),
        HumanMessage(content="Current question", id="h3"),
    ]

    assert (
        middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())
        is None
    )
    assert (
        middleware.before_agent({"messages": messages[:-1]}, runtime=SimpleNamespace())
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
