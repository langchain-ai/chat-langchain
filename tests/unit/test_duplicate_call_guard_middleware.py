"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware


def _request(
    name: str,
    call_id: str,
    args: dict,
    messages: list | None = None,
    content: str = "Question",
):
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state={"messages": [HumanMessage(content=content), *(messages or [])]},
        runtime=None,
    )


def _history(request, result):
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": request.tool_call["name"],
                    "args": request.tool_call["args"],
                    "id": request.tool_call["id"],
                    "type": "tool_call",
                }
            ],
        ),
        result,
    ]


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
        history = _history(
            _request("search_docs", "call-1", {"query": "middleware"}), first
        )
        second = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "middleware"}, history), handler
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
        first_request = _request("search_docs", "call-1", {"query": "first"})
        first = await middleware.awrap_tool_call(first_request, handler)
        history = _history(first_request, first)
        second_request = _request("search_docs", "call-2", {"query": "second"}, history)
        await middleware.awrap_tool_call(second_request, handler)

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
        first_request = _request(
            "check_links", "call-1", {"urls": ["https://one.example"]}
        )
        first = await asyncio.create_task(
            middleware.awrap_tool_call(first_request, handler)
        )
        history = _history(first_request, first)
        second_request = _request(
            "check_links", "call-2", {"urls": ["https://two.example"]}, history
        )
        second = await asyncio.create_task(
            middleware.awrap_tool_call(second_request, handler)
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert "may only be called once per turn" in second.content


def test_tool_call_safety_cap_stops_after_eight_calls():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content=f"result-{len(calls)}",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        history = []
        results = []
        for index in range(9):
            request = _request(
                "search_docs", f"call-{index}", {"query": str(index)}, history
            )
            result = await asyncio.create_task(
                middleware.awrap_tool_call(request, handler)
            )
            results.append(result)
            history.extend(_history(request, result))
        return results

    results = asyncio.run(invoke())

    assert len(calls) == 8
    assert "Stop calling this tool" in results[-1].content
