"""Tests for per-turn duplicate tool-call suppression."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware


def _request(
    name: str,
    call_id: str,
    args: dict,
    messages: list | None = None,
    run_id: str = "run-1",
):
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state={"messages": messages or [HumanMessage(content="Question")]},
        runtime=SimpleNamespace(config={"run_id": run_id}),
    )


def test_identical_call_returns_cached_content_with_current_call_identity():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    messages = [HumanMessage(content="Question")]

    async def handler(request):
        calls.append(request)
        result = ToolMessage(
            content="cached result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="success",
        )
        messages.extend([AIMessage(content="", tool_calls=[request.tool_call]), result])
        return result

    async def invoke_once(call_id: str, run_id: str):
        return await middleware.awrap_tool_call(
            _request(
                "search_docs",
                call_id,
                {"query": "middleware"},
                messages,
                run_id,
            ),
            handler,
        )

    async def invoke():
        first = await asyncio.create_task(invoke_once("call-1", "run-1"))
        second = await asyncio.create_task(invoke_once("call-2", "run-2"))
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
            status="success",
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
            status="success",
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
    messages = [HumanMessage(content="Question")]

    async def handler(request):
        calls.append(request)
        result = ToolMessage(
            content="validated",
            name="check_links",
            tool_call_id=request.tool_call["id"],
            status="success",
        )
        messages.extend([AIMessage(content="", tool_calls=[request.tool_call]), result])
        return result

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-1",
                {"urls": ["https://one.example"]},
                messages,
            ),
            handler,
        )
        second = await middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-2",
                {"urls": ["https://two.example"]},
                messages,
            ),
            handler,
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert "may only be called once per turn" in second.content


def test_new_human_message_resets_duplicate_and_check_links_state():
    middleware = DuplicateCallGuardMiddleware()
    messages = [HumanMessage(content="First question")]
    calls = []

    async def handler(request):
        calls.append(request)
        result = ToolMessage(
            content="validated",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="success",
        )
        messages.extend([AIMessage(content="", tool_calls=[request.tool_call]), result])
        return result

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "same"}, messages),
            handler,
        )
        duplicate = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "same"}, messages),
            handler,
        )
        first_links = await middleware.awrap_tool_call(
            _request("check_links", "call-3", {"urls": ["https://one.example"]}, messages),
            handler,
        )
        messages.append(HumanMessage(content="Second question"))
        second = await middleware.awrap_tool_call(
            _request("search_docs", "call-4", {"query": "same"}, messages),
            handler,
        )
        second_links = await middleware.awrap_tool_call(
            _request("check_links", "call-5", {"urls": ["https://one.example"]}, messages),
            handler,
        )
        refused_links = await middleware.awrap_tool_call(
            _request("check_links", "call-6", {"urls": ["https://two.example"]}, messages),
            handler,
        )
        return first, duplicate, first_links, second, second_links, refused_links

    first, duplicate, first_links, second, second_links, refused_links = asyncio.run(invoke())

    assert len(calls) == 4
    assert first.content == "validated"
    assert "already made on this turn" in duplicate.content
    assert first_links.content == "validated"
    assert second.content == "validated"
    assert second_links.content == "validated"
    assert "may only be called once per turn" in refused_links.content
