"""Tests for ingress input caps, turn cleanup, and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
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


def test_before_agent_removes_superseded_human_and_truncates_latest():
    middleware = IngressGuardsMiddleware()
    history = [
        HumanMessage(content="Earlier answered question", id="h1"),
        AIMessage(content="Earlier answer", id="a1"),
    ]
    superseded = HumanMessage(content="Abandoned question", id="h2")
    latest = HumanMessage(content="x" * (MAX_MESSAGE_CHARS + 50), id="h3")
    state = {"messages": [*history, superseded, latest]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    removals = [
        message for message in update["messages"] if isinstance(message, RemoveMessage)
    ]
    assert [message.id for message in removals] == ["h2"]
    retained = add_messages(state["messages"], update["messages"])
    assert retained == [*history, latest]
    assert retained[-1].id == "h3"
    assert len(retained[-1].content) == MAX_MESSAGE_CHARS


def test_before_agent_noop_with_single_trailing_human_and_tool():
    middleware = IngressGuardsMiddleware()
    state = {
        "messages": [
            AIMessage(content="Earlier answer", id="a1"),
            HumanMessage(content="Current question", id="h1"),
            ToolMessage(content="Tool result", tool_call_id="call1", id="t1"),
        ]
    }

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_before_agent_removes_multiple_humans_without_ai():
    middleware = IngressGuardsMiddleware()
    latest = HumanMessage(content="Current question", id="h3")
    tool = ToolMessage(content="Tool result", tool_call_id="call1", id="t1")
    state = {
        "messages": [
            HumanMessage(content="First abandoned question", id="h1"),
            tool,
            HumanMessage(content="Second abandoned question", id="h2"),
            latest,
        ]
    }

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert all(isinstance(message, RemoveMessage) for message in update["messages"])
    assert {message.id for message in update["messages"]} == {"h1", "h2"}
    assert add_messages(state["messages"], update["messages"]) == [tool, latest]


@pytest.mark.parametrize("with_history", [False, True])
def test_first_model_call_sees_only_latest_unanswered_question(with_history):
    history = (
        [
            HumanMessage(content="What is LangGraph?", id="h0"),
            AIMessage(content="A framework for stateful agents.", id="a0"),
        ]
        if with_history
        else []
    )
    latest = HumanMessage(
        content="How does a checkpointer differ from a store? When should I use each?",
        id="h3",
    )
    model_inputs = []

    class CaptureModelInputs(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            model_inputs.extend(messages)

    model = FakeMessagesListChatModel(responses=[AIMessage(content="Test response")])
    agent = create_agent(model, middleware=[IngressGuardsMiddleware()])

    agent.invoke(
        {
            "messages": [
                *history,
                HumanMessage(content="How do I stream responses?", id="h1"),
                HumanMessage(content="How do I persist state?", id="h2"),
                latest,
            ]
        },
        config={"callbacks": [CaptureModelInputs()]},
    )

    assert model_inputs == [[*history, latest]]


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
