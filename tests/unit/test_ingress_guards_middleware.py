"""Tests for ingress input caps, orphan cleanup, and root-trace metadata helpers."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
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


@pytest.mark.parametrize(
    "orphan_content",
    ["How do I deploy a graph?", "How do I stream graph outputs?"],
)
@pytest.mark.parametrize(
    "current_content",
    [
        "How do I stream graph outputs? Use Python only.",
        [{"type": "text", "text": "How do I stream graph outputs? Use Python only."}],
    ],
)
def test_before_agent_removes_orphans_and_preserves_latest(
    orphan_content, current_content
):
    middleware = IngressGuardsMiddleware()
    history = [
        HumanMessage(content="What is LangGraph?", id="answered"),
        AIMessage(content="A graph framework.", id="reply"),
    ]
    orphans = [
        HumanMessage(content=orphan_content, id="orphan-1"),
        HumanMessage(content=orphan_content, id="orphan-2"),
    ]
    current = HumanMessage(content=current_content, id="current")
    original = current.model_dump()
    messages = history + orphans + [current]

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert all(isinstance(message, RemoveMessage) for message in update["messages"])
    assert {message.id for message in update["messages"]} == {"orphan-1", "orphan-2"}
    assert add_messages(messages, update["messages"]) == history + [current]
    assert current.model_dump() == original


def test_before_agent_removes_orphan_and_caps_latest_in_same_update():
    messages = [
        HumanMessage(content="Old question", id="orphan"),
        HumanMessage(content="x" * (MAX_MESSAGE_CHARS + 50), id="current"),
    ]

    update = IngressGuardsMiddleware().before_agent(
        {"messages": messages}, runtime=SimpleNamespace()
    )

    assert update is not None
    remaining = add_messages(messages, update["messages"])
    assert len(remaining) == 1
    assert remaining[0].id == "current"
    assert remaining[0].content == "x" * MAX_MESSAGE_CHARS


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [HumanMessage(content="Current question", id="current")],
        [
            HumanMessage(content="Answered question", id="answered"),
            AIMessage(content="Answer", id="reply"),
            HumanMessage(content="Current question", id="current"),
        ],
        [
            HumanMessage(content="Earlier context", id="earlier"),
            SystemMessage(content="Context boundary", id="system"),
            HumanMessage(content="Current question", id="current"),
        ],
        [
            HumanMessage(content="First question", id="first"),
            HumanMessage(content="Second question", id="second"),
            AIMessage(content="Answer", id="reply"),
        ],
    ],
)
def test_before_agent_leaves_messages_without_trailing_orphans_unchanged(messages):
    original = [message.model_dump() for message in messages]

    assert (
        IngressGuardsMiddleware().before_agent(
            {"messages": messages}, runtime=SimpleNamespace()
        )
        is None
    )
    assert [message.model_dump() for message in messages] == original


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "orphan_content",
    [None, "How do I deploy a graph?", "How do I stream graph outputs?"],
)
def test_agent_model_receives_latest_question_without_orphans(
    async_mode, orphan_content
):
    seen_messages = []

    @before_model
    def capture_messages(state, runtime):
        seen_messages.extend(state["messages"])

    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Use Python streaming.")])
    )
    agent = create_agent(
        model=model,
        middleware=[IngressGuardsMiddleware(), capture_messages],
        checkpointer=InMemorySaver(),
    )
    history = [
        HumanMessage(content="What is LangGraph?", id="answered"),
        AIMessage(content="A graph framework.", id="reply"),
    ]
    current = HumanMessage(
        content="How do I stream graph outputs? Use Python only.", id="current"
    )
    messages = list(history)
    if orphan_content is not None:
        messages.append(HumanMessage(content=orphan_content, id="orphan"))
    config = {"configurable": {"thread_id": "interrupted-thread"}}
    agent.update_state(config, {"messages": messages})

    if async_mode:
        result = asyncio.run(agent.ainvoke({"messages": [current]}, config))
    else:
        result = agent.invoke({"messages": [current]}, config)

    assert seen_messages == history + [current]
    assert result["messages"][:-1] == history + [current]
    assert result["messages"][-1].content == "Use Python streaming."


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
