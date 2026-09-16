"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.duplicate_call_guard_middleware import (
    DuplicateCallGuardMiddleware,
    consume_repair_budget,
)


def _request(name: str, call_id: str, args: dict, content: str = "Question"):
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state={"messages": [HumanMessage(content=content)]},
        runtime=None,
    )


def test_identical_call_returns_cached_content_with_current_call_identity():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="cached result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "middleware"}), handler
        )
        second = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "middleware"}), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "cached result"
    assert second.tool_call_id == "call-2"
    assert second.name == "search_docs"
    assert "already made on this turn" in second.content
    assert "cached result" in second.content


def test_different_arguments_pass_through_for_non_budgeted_tools():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content=request.tool_call["args"]["query"],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "first"}), handler
        )
        await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "second"}), handler
        )

    asyncio.run(invoke())

    assert len(calls) == 2


def test_failed_call_is_not_cached_and_can_be_retried():
    middleware = DuplicateCallGuardMiddleware()
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return ToolMessage(
            content="success",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        request = _request("search_docs", "call-1", {"query": "retry"})
        with pytest.raises(RuntimeError):
            await middleware.awrap_tool_call(request, handler)
        return await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "retry"}), handler
        )

    result = asyncio.run(invoke())

    assert attempts == 2
    assert result.content == "success"


def test_check_links_allows_one_invocation_per_turn():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="validated",
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request("check_links", "call-1", {"urls": ["https://one.example"]}),
            handler,
        )
        second = await middleware.awrap_tool_call(
            _request("check_links", "call-2", {"urls": ["https://two.example"]}),
            handler,
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert "may only be called once per turn" in second.content


def test_separate_asyncio_tasks_share_turn_budget_and_cache():
    middleware = DuplicateCallGuardMiddleware()
    state = {"messages": [HumanMessage(id="turn-1", content="Question")]}
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0)
        return ToolMessage(
            content="validated",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke(request):
        return await middleware.awrap_tool_call(request, handler)

    async def run():
        requests = [
            ToolCallRequest(
                tool_call={
                    "name": "check_links",
                    "id": f"call-{index}",
                    "args": {"urls": ["https://one.example"]},
                },
                tool=None,
                state=state,
                runtime=None,
            )
            for index in range(2)
        ]
        return await asyncio.gather(*(invoke(request) for request in requests))

    first, second = asyncio.run(run())

    assert len(calls) == 1
    assert "validated" in first.content
    assert "may only be called once per turn" in second.content


def test_repair_budget_is_shared_by_model_guards_for_one_human_message():
    messages = [HumanMessage(id="turn-2", content="Question")]
    state = {"messages": messages}

    assert consume_repair_budget(state, messages)
    assert consume_repair_budget(state, messages)
    assert not consume_repair_budget(state, messages)
