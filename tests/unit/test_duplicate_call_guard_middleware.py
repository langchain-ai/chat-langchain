"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from src.middleware import citation_guard_middleware
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


def _apply_result(result, state):
    if isinstance(result, Command):
        state.update(
            {key: value for key, value in result.update.items() if key != "messages"}
        )
        return result.update["messages"][0]
    return result


def test_identical_call_returns_cached_content_with_current_call_identity():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    state = {"messages": [HumanMessage(content="Question")]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="cached result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "middleware"}, state=state),
            handler,
        )
        first_message = _apply_result(first, state)
        second = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "middleware"}, state=state),
            handler,
        )
        return first_message, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "cached result"
    assert second.tool_call_id == "call-2"
    assert second.name == "search_docs"
    assert "already made on this turn" in second.content
    assert "cached result" in second.content


def test_identical_calls_in_separate_tasks_share_agent_state():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    state = {"messages": [HumanMessage(content="Question")]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="cached result", tool_call_id=request.tool_call["id"]
        )

    async def invoke():
        first = await asyncio.create_task(
            middleware.awrap_tool_call(
                _request("search_docs", "call-1", {"query": "middleware"}, state=state),
                handler,
            )
        )
        _apply_result(first, state)
        second = await asyncio.create_task(
            middleware.awrap_tool_call(
                _request("search_docs", "call-2", {"query": "middleware"}, state=state),
                handler,
            )
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert isinstance(first, Command)
    assert len(calls) == 1
    assert _apply_result(first, state).content == "cached result"
    assert "already made on this turn" in second.content
    assert "cached result" in second.content


def test_different_arguments_pass_through_for_non_budgeted_tools():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    state = {"messages": [HumanMessage(content="Question")]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content=request.tool_call["args"]["query"],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "first"}, state=state), handler
        )
        _apply_result(first, state)
        second = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "second"}, state=state), handler
        )
        _apply_result(second, state)

    asyncio.run(invoke())

    assert len(calls) == 2


def test_failed_call_is_not_cached_and_can_be_retried():
    middleware = DuplicateCallGuardMiddleware()
    attempts = 0
    state = {"messages": [HumanMessage(content="Question")]}

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
        request = _request("search_docs", "call-1", {"query": "retry"}, state=state)
        with pytest.raises(RuntimeError):
            await middleware.awrap_tool_call(request, handler)
        result = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "retry"}, state=state), handler
        )
        return _apply_result(result, state)

    result = asyncio.run(invoke())

    assert attempts == 2
    assert result.content == "success"


def test_check_links_allows_one_invocation_per_turn():
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

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request(
                "check_links", "call-1", {"urls": ["https://one.example"]}, state=state
            ),
            handler,
        )
        first_message = _apply_result(first, state)
        second = await middleware.awrap_tool_call(
            _request(
                "check_links", "call-2", {"urls": ["https://two.example"]}, state=state
            ),
            handler,
        )
        return first_message, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert "may only be called once per turn" in second.content


def test_citation_retry_message_does_not_reset_check_links_budget():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    state = {"messages": [HumanMessage(content="Question")]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request(
                "check_links", "call-1", {"urls": ["https://one.example"]}, state=state
            ),
            handler,
        )
        _apply_result(first, state)
        state["messages"].append(
            HumanMessage(content=citation_guard_middleware._RETRY_INSTRUCTIONS)
        )
        return await middleware.awrap_tool_call(
            _request(
                "check_links", "call-2", {"urls": ["https://two.example"]}, state=state
            ),
            handler,
        )

    result = asyncio.run(invoke())

    assert len(calls) == 1
    assert "may only be called once per turn" in result.content
