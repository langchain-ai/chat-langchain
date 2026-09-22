"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware


def _request(
    name: str,
    call_id: str,
    args: dict,
    content: str = "Question",
    state: dict | None = None,
):
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state=state or {"messages": [HumanMessage(content=content)]},
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
        state = {"messages": [HumanMessage(content="Question")]}
        first = await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "middleware"}, state=state), handler
        )
        second = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "middleware"}, state=state), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert isinstance(first, Command)
    assert first.update["messages"][0].content == "cached result"
    assert isinstance(second, Command)
    second_message = second.update["messages"][0]
    assert second_message.tool_call_id == "call-2"
    assert second_message.name == "search_docs"
    assert "already made on this turn" in second_message.content
    assert "cached result" in second_message.content


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
    assert isinstance(result, Command)
    assert result.update["messages"][0].content == "success"


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
        state = {"messages": [HumanMessage(content="Question")]}
        first = await middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-1",
                {"urls": ["https://one.example"]},
                state=state,
            ),
            handler,
        )
        second = await middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-2",
                {"urls": ["https://two.example"]},
                state=state,
            ),
            handler,
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert isinstance(first, Command)
    assert first.update["messages"][0].content == "validated"
    assert isinstance(second, Command)
    assert "may only be called once per turn" in second.update["messages"][0].content


def test_identical_check_links_calls_in_separate_tasks_share_turn_state():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    state = {"messages": [HumanMessage(content="Question")]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="validated",
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke(request):
        return await middleware.awrap_tool_call(request, handler)

    async def run_in_separate_tasks():
        first = await asyncio.create_task(
            invoke(
                _request(
                    "check_links",
                    "call-1",
                    {"urls": ["https://one.example"]},
                    state=state,
                )
            )
        )
        second = await asyncio.create_task(
            invoke(
                _request(
                    "check_links",
                    "call-2",
                    {"urls": ["https://two.example"]},
                    state=state,
                )
            )
        )
        return first, second

    first, second = asyncio.run(run_in_separate_tasks())

    assert isinstance(first, Command)
    assert isinstance(second, Command)
    assert len(calls) == 1
    assert "may only be called once per turn" in second.update["messages"][0].content


def test_tool_budget_returns_explicit_exhaustion_message():
    middleware = DuplicateCallGuardMiddleware()
    state = {"messages": [HumanMessage(content="Question")]}

    async def handler(request):
        return ToolMessage(content="ok", name="search_docs", tool_call_id=request.tool_call["id"])

    async def invoke():
        for index in range(12):
            await middleware.awrap_tool_call(
                _request("search_docs", f"call-{index}", {"query": str(index)}, state=state),
                handler,
            )
        return await middleware.awrap_tool_call(
            _request("search_docs", "call-12", {"query": "12"}, state=state), handler
        )

    result = asyncio.run(invoke())

    assert isinstance(result, Command)
    assert "tool budget is exhausted" in result.update["messages"][0].content
