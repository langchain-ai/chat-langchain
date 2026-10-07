"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
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


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "messages, orphan_indices",
    [
        ([], []),
        ([HumanMessage(content="Current")], []),
        (
            [
                HumanMessage(content="A"),
                AIMessage(content="Reply"),
                HumanMessage(content="Current"),
            ],
            [],
        ),
        (
            [
                HumanMessage(content="A"),
                AIMessage(content="Reply"),
                HumanMessage(content="B"),
                HumanMessage(content="Current"),
            ],
            [2],
        ),
        (
            [
                HumanMessage(content="A"),
                HumanMessage(content="B"),
                HumanMessage(content="Current"),
            ],
            [0, 1],
        ),
        (
            [
                HumanMessage(content="A"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_docs", "args": {}, "id": "search"}],
                ),
                ToolMessage(content="Docs", tool_call_id="search"),
                HumanMessage(content="Current"),
            ],
            [],
        ),
        (
            [
                HumanMessage(content="A"),
                ToolMessage(content="Docs", tool_call_id="search"),
                HumanMessage(content="Current"),
            ],
            [],
        ),
        (
            [
                HumanMessage(content="A"),
                HumanMessage(content="Current"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_docs", "args": {}, "id": "search"}],
                ),
                ToolMessage(content="Docs", tool_call_id="search"),
            ],
            [0],
        ),
        (
            [
                HumanMessage(content="A"),
                SystemMessage(content="Context"),
                HumanMessage(content="Current"),
            ],
            [],
        ),
    ],
)
def test_model_call_marks_only_adjacent_orphans_without_mutating_state(
    async_call, messages, orphan_indices
):
    middleware = IngressGuardsMiddleware()
    state = {"messages": messages}
    original_state = deepcopy(state)
    request = ModelRequest(model=object(), messages=messages, state=state)
    response = ModelResponse(result=[AIMessage(content="Answer")])
    calls = []

    def handler(model_request):
        calls.append(model_request)
        return response

    async def async_handler(model_request):
        return handler(model_request)

    if async_call:
        result = asyncio.run(middleware.awrap_model_call(request, async_handler))
    else:
        result = middleware.wrap_model_call(request, handler)

    assert result is response
    assert len(calls) == 1
    forwarded = calls[0]
    expected = []
    for index, message in enumerate(messages):
        expected.append(message)
        if index in orphan_indices:
            expected.append(
                AIMessage(
                    content="(No answer was produced for this message because the run was interrupted.)"
                )
            )
    assert forwarded.messages == expected
    assert middleware._mark_orphaned_turns(forwarded) is forwarded
    assert forwarded.state is state
    assert state == original_state
    assert state["messages"] is messages
    assert request.messages is messages
    assert (forwarded is request) == (not orphan_indices)
    latest_human = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    for guard in (CitationGuardMiddleware(), DocsResearchGuardMiddleware()):
        latest_index = guard._latest_human_index(forwarded.messages)
        if latest_human is not None:
            assert latest_index == expected.index(latest_human)
            assert forwarded.messages[latest_index] is latest_human
        else:
            assert latest_index == -1


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
