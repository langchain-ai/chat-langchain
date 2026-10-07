"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
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


def _answered_turn(content="Hello"):
    return [
        HumanMessage(content=content, id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"name": "check_links", "args": {}, "id": "call1"}],
        ),
        ToolMessage(content="Valid", id="t1", tool_call_id="call1"),
        AIMessage(content="Previous answer", id="a2"),
    ]


def test_before_agent_removes_all_messages_after_latest_human():
    middleware = IngressGuardsMiddleware()
    history = [HumanMessage(content="Earlier question", id="h0")]
    messages = history + _answered_turn()

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update == {
        "messages": [
            RemoveMessage(id="a1"),
            RemoveMessage(id="t1"),
            RemoveMessage(id="a2"),
        ]
    }
    assert add_messages(messages, update["messages"]) == messages[:2]


def test_before_agent_truncates_resent_message_and_removes_stale_replies():
    middleware = IngressGuardsMiddleware()
    messages = _answered_turn("x" * (MAX_MESSAGE_CHARS + 50))

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][:3] == [
        RemoveMessage(id="a1"),
        RemoveMessage(id="t1"),
        RemoveMessage(id="a2"),
    ]
    result = add_messages(messages, update["messages"])
    assert result == [HumanMessage(content="x" * MAX_MESSAGE_CHARS, id="h1")]


def test_before_agent_preserves_non_text_blocks_when_capping_resent_message():
    middleware = IngressGuardsMiddleware()
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    messages = _answered_turn(
        ["x" * (MAX_MESSAGE_CHARS - 5), image, {"type": "text", "text": "y" * 10}]
    )

    update = middleware.before_agent({"messages": messages}, runtime=SimpleNamespace())

    assert update is not None
    result = add_messages(messages, update["messages"])
    assert len(result) == 1
    assert result[0].content == [
        "x" * (MAX_MESSAGE_CHARS - 5),
        image,
        {"type": "text", "text": "y" * 5},
    ]


def test_resent_human_id_reaches_primary_model_without_stale_replies():
    requests = []

    @wrap_model_call
    def capture_request(request, handler):
        requests.append(request)
        return handler(request)

    model = GenericFakeChatModel(messages=iter([AIMessage(content="New answer")]))
    agent = create_agent(
        model,
        middleware=[IngressGuardsMiddleware(), capture_request],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "resent-turn"}}
    agent.update_state(config, {"messages": _answered_turn()})
    resent = HumanMessage(content="Hello again", id="h1")

    result = agent.invoke({"messages": [resent]}, config)

    assert len(requests) == 1
    assert requests[0].model is model
    assert requests[0].messages == [resent]
    assert result["messages"][0] == resent
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "New answer"


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
