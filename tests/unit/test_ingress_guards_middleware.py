"""Tests for ingress guards input caps, current-turn focus, and trace metadata."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

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


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "messages, marked_indices",
    [
        (
            [
                HumanMessage(content="Q1"),
                AIMessage(content="A1"),
                HumanMessage(content="Q2", id="q2", name="user"),
                HumanMessage(content="Q3. Two sentences."),
            ],
            [2],
        ),
        (
            [
                HumanMessage(content="Q1"),
                HumanMessage(content="Q2"),
                HumanMessage(content="Q3"),
            ],
            [0, 1],
        ),
        (
            [
                HumanMessage(content="Q1"),
                AIMessage(content="A1"),
                HumanMessage(content="Q2"),
            ],
            [],
        ),
        ([HumanMessage(content="Q1")], []),
        ([SystemMessage(content="Instructions")], []),
        ([], []),
        (
            [
                HumanMessage(content="Q1"),
                HumanMessage(content="Q2"),
                AIMessage(
                    content="Searching",
                    tool_calls=[{"name": "search", "args": {}, "id": "call1"}],
                ),
                ToolMessage(content="Docs", tool_call_id="call1"),
            ],
            [0],
        ),
    ],
)
def test_model_call_marks_only_superseded_messages(use_async, messages, marked_indices):
    middleware = IngressGuardsMiddleware()
    state = {"messages": messages}
    request = ModelRequest(model=object(), messages=messages, state=state)
    originals = [message.model_copy(deep=True) for message in messages]
    response = ModelResponse(result=[AIMessage(content="Answer")])
    calls = []

    def handler(model_request):
        calls.append(model_request)
        return response

    async def async_handler(model_request):
        return handler(model_request)

    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if use_async
        else middleware.wrap_model_call(request, handler)
    )

    assert result is response
    assert len(calls) == 1
    assert request.messages == originals
    assert state["messages"] == originals
    for index, message in enumerate(calls[0].messages):
        if index in marked_indices:
            assert message.content == (
                "[Earlier message with no reply - superseded by the user's next message] "
                + originals[index].content
            )
            assert message.id == originals[index].id
            assert message.name == originals[index].name
            assert message is not messages[index]
        else:
            assert message is messages[index]
    if not marked_indices:
        assert calls[0] is request


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        ["Earlier question"],
        [{"type": "text", "text": "Earlier question", "metadata": {"source": "user"}}],
        [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        [
            "Earlier question",
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ],
        [],
    ],
)
def test_model_call_marks_block_content_without_mutating_state(use_async, content):
    middleware = IngressGuardsMiddleware()
    earlier = HumanMessage(content=content)
    latest = HumanMessage(content=[{"type": "text", "text": "Latest question"}])
    state = {"messages": [earlier, latest]}
    request = ModelRequest(model=object(), messages=state["messages"], state=state)
    originals = [message.model_copy(deep=True) for message in request.messages]
    calls = []
    response = ModelResponse(result=[AIMessage(content="Answer")])

    def handler(model_request):
        calls.append(model_request)
        return response

    async def async_handler(model_request):
        return handler(model_request)

    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if use_async
        else middleware.wrap_model_call(request, handler)
    )

    assert result is response
    assert calls[0].messages[0].content == [
        {
            "type": "text",
            "text": "[Earlier message with no reply - superseded by the user's next message] ",
        },
        *content,
    ]
    assert calls[0].messages[-1] is latest
    assert request.messages == originals
    assert state["messages"] == originals


def test_model_call_keeps_latest_ingress_truncation():
    middleware = IngressGuardsMiddleware()
    earlier = HumanMessage(content="Earlier question")
    latest = HumanMessage(content="x" * (MAX_MESSAGE_CHARS + 50), id="latest")
    state = {"messages": [earlier, latest]}
    middleware.before_agent(state, runtime=SimpleNamespace())
    request = ModelRequest(model=object(), messages=state["messages"], state=state)
    calls = []

    def handler(model_request):
        calls.append(model_request)
        return ModelResponse(result=[AIMessage(content="Answer")])

    middleware.wrap_model_call(request, handler)

    assert calls[0].messages[-1] is latest
    assert latest.content == "x" * MAX_MESSAGE_CHARS
    assert earlier.content == "Earlier question"


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
