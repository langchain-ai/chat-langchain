"""Tests for ingress guards input caps and root-trace metadata helpers."""

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


@pytest.mark.parametrize("prefix_length, trailing_count", [(0, 2), (2, 2), (0, 3)])
def test_before_agent_repairs_stranded_human_messages(prefix_length, trailing_count):
    middleware = IngressGuardsMiddleware()
    prefix = [
        HumanMessage(content="Answered question", id="answered-human"),
        AIMessage(content="Answer", id="answered-ai"),
    ][:prefix_length]
    trailing = [
        HumanMessage(
            content=f"Question {index}",
            id=f"human-{index}",
            additional_kwargs={"custom": index},
        )
        for index in range(trailing_count)
    ]
    messages = prefix + trailing

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    repaired = add_messages(messages, update["messages"])
    assert repaired[:prefix_length] == prefix
    assert repaired[-1] == trailing[-1]
    assert len(repaired) == len(messages) + trailing_count - 1
    marker_ids = []
    for index, human in enumerate(trailing[:-1]):
        human_index = prefix_length + index * 2
        assert repaired[human_index] == human
        marker = repaired[human_index + 1]
        assert isinstance(marker, AIMessage)
        assert (
            marker.content
            == "[Previous request was stopped before an answer was produced.]"
        )
        assert marker.id is not None
        marker_ids.append(marker.id)
    assert len(set(marker_ids + [message.id for message in messages])) == len(repaired)
    assert (
        middleware.before_agent({"messages": repaired}, runtime=SimpleNamespace())
        is None
    )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [SystemMessage(content="Instructions", id="system")],
        [
            HumanMessage(content="Answered question", id="h1"),
            AIMessage(content="Answer", id="a1"),
            HumanMessage(content="Latest question", id="h2"),
        ],
        [
            HumanMessage(content="Question", id="h1"),
            HumanMessage(content="Follow-up", id="h2"),
            AIMessage(content="Answer", id="a1"),
        ],
        [
            HumanMessage(content="Question", id="h1"),
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{"name": "search", "args": {}, "id": "t1"}],
            ),
            ToolMessage(content="Result", tool_call_id="t1", id="tool"),
        ],
    ],
)
def test_before_agent_leaves_normal_history_unchanged(messages):
    middleware = IngressGuardsMiddleware()

    assert (
        middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())
        is None
    )


def test_before_agent_caps_latest_message_while_repairing_history():
    middleware = IngressGuardsMiddleware()
    stranded = HumanMessage(content="Unanswered question", id="h1")
    latest = HumanMessage(content="x" * (MAX_MESSAGE_CHARS + 50), id="h2")
    messages = [stranded, latest]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    repaired = add_messages(messages, update["messages"])
    assert [message.type for message in repaired] == ["human", "ai", "human"]
    assert repaired[0] == stranded
    assert repaired[-1].id == latest.id
    assert repaired[-1].content == "x" * MAX_MESSAGE_CHARS


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
