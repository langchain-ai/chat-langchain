"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
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


def test_before_agent_preserves_answered_messages():
    middleware = IngressGuardsMiddleware()
    messages = [
        HumanMessage(content="Earlier question", id="h1"),
        AIMessage(content="Earlier answer", id="a1"),
        HumanMessage(content="Latest question", id="h2"),
    ]

    assert (
        middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())
        is None
    )
    assert middleware.before_agent({"messages": []}, runtime=SimpleNamespace()) is None


def test_before_agent_merges_stale_message_and_preserves_latest_text():
    middleware = IngressGuardsMiddleware()
    latest_text = "  Latest question?\nKeep `this` verbatim.  "
    messages = [
        HumanMessage(content="Stopped question", id="h1"),
        HumanMessage(content=latest_text, id="h2"),
    ]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert isinstance(update["messages"][0], RemoveMessage)
    assert update["messages"][0].id == "h1"
    merged = add_messages(messages, update["messages"])
    assert len(merged) == 1
    assert merged[0].id == "h2"
    assert merged[0].content.startswith("[Stopped-turn context]\n- Stopped question")
    assert (
        "Answer the latest message; earlier messages were not answered"
        in merged[0].content
    )
    assert merged[0].content.endswith("\n\n" + latest_text)
    assert messages[-1].content == latest_text


@pytest.mark.parametrize(
    "stale_content, latest_content",
    [
        ("Same question", "Same question"),
        ([{"type": "text", "text": "Same "}, "question"], "Same question"),
        ("Same question", [{"type": "text", "text": "Same question"}]),
    ],
)
def test_before_agent_removes_identical_resubmission_without_marker(
    stale_content, latest_content
):
    middleware = IngressGuardsMiddleware()
    messages = [
        HumanMessage(content=stale_content, id="h1"),
        HumanMessage(content=latest_content, id="h2"),
    ]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    merged = add_messages(messages, update["messages"])
    assert len(merged) == 1
    assert merged[0].id == "h2"
    assert merged[0].content == latest_content


def test_before_agent_preserves_list_content_and_non_text_blocks():
    middleware = IngressGuardsMiddleware()
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    latest_content = ["Latest ", {"type": "text", "text": "question?"}, image]
    messages = [
        HumanMessage(
            content=["Stopped ", {"type": "text", "text": "question"}, image],
            id="h1",
        ),
        HumanMessage(content=latest_content, id="h2"),
    ]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    merged = add_messages(messages, update["messages"])
    assert len(merged) == 1
    assert (
        merged[0]
        .content[0]["text"]
        .startswith("[Stopped-turn context]\n- Stopped question")
    )
    assert "image_url" not in merged[0].content[0]["text"]
    assert merged[0].content[1:] == latest_content


def test_before_agent_removes_only_unanswered_messages_in_order():
    middleware = IngressGuardsMiddleware()
    messages = [
        HumanMessage(content="Answered question", id="h0"),
        AIMessage(content="Answer", id="a0"),
        HumanMessage(content="x" * 600, id="h1"),
        SystemMessage(content="Thread metadata", id="s1"),
        HumanMessage(content="Latest question", id="h2"),
        HumanMessage(content="Another stopped question", id="h3"),
        HumanMessage(content="Latest question", id="h4"),
    ]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert [message.id for message in update["messages"]] == ["h1", "h2", "h3", "h4"]
    merged = add_messages(messages, update["messages"])
    assert [message.id for message in merged] == ["h0", "a0", "s1", "h4"]
    assert "- " + "x" * 500 + "\n- Another stopped question" in merged[-1].content
    assert "x" * 501 not in merged[-1].content
    assert "- Latest question" not in merged[-1].content


def test_before_agent_caps_list_content_when_reconciling():
    middleware = IngressGuardsMiddleware()
    latest_content = ["x" * MAX_MESSAGE_CHARS, {"type": "text", "text": "overflow"}]
    messages = [
        HumanMessage(content="Stopped question", id="h1"),
        HumanMessage(content=latest_content, id="h2"),
    ]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][-1].content[1:] == [
        "x" * MAX_MESSAGE_CHARS,
        {"type": "text", "text": ""},
    ]


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
