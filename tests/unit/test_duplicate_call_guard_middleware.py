"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.duplicate_call_guard_middleware import (
    _CHECK_LINKS_REFUSAL,
    _DUPLICATE_NOTE,
    DuplicateCallGuardMiddleware,
)


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
        state={"messages": messages or [HumanMessage(content=content)]},
        runtime=None,
    )


def _history(name: str, call_id: str, args: dict, result: ToolMessage):
    return [
        HumanMessage(content="Question"),
        AIMessage(
            content="",
            tool_calls=[{"name": name, "id": call_id, "args": args}],
        ),
        result,
    ]


def test_identical_check_links_calls_in_separate_tasks_are_refused():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    args = {"urls": ["https://one.example"]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="validated",
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first_request = _request("check_links", "call-1", args)
        first = await asyncio.create_task(
            middleware.awrap_tool_call(first_request, handler)
        )
        second_request = _request(
            "check_links",
            "call-2",
            args,
            _history("check_links", "call-1", args, first),
        )
        second = await asyncio.create_task(
            middleware.awrap_tool_call(second_request, handler)
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert second.content == _CHECK_LINKS_REFUSAL


def test_identical_non_check_links_calls_in_separate_tasks_return_cached_content():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    args = {"query": "middleware"}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="cached result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first_request = _request("search_docs", "call-1", args)
        first = await asyncio.create_task(
            middleware.awrap_tool_call(first_request, handler)
        )
        second_request = _request(
            "search_docs",
            "call-2",
            args,
            _history("search_docs", "call-1", args, first),
        )
        second = await asyncio.create_task(
            middleware.awrap_tool_call(second_request, handler)
        )
        return second

    second = asyncio.run(invoke())

    assert len(calls) == 1
    assert second.tool_call_id == "call-2"
    assert second.name == "search_docs"
    assert _DUPLICATE_NOTE in second.content
    assert "cached result" in second.content


def test_different_arguments_in_separate_task_reach_handler():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    first_args = {"query": "first"}
    second_args = {"query": "second"}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content=request.tool_call["args"]["query"],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first_request = _request("search_docs", "call-1", first_args)
        first = await asyncio.create_task(
            middleware.awrap_tool_call(first_request, handler)
        )
        second_request = _request(
            "search_docs",
            "call-2",
            second_args,
            _history("search_docs", "call-1", first_args, first),
        )
        second = await asyncio.create_task(
            middleware.awrap_tool_call(second_request, handler)
        )
        return second

    second = asyncio.run(invoke())

    assert len(calls) == 2
    assert second.content == "second"


def test_failed_tool_result_is_not_cached_and_can_be_retried():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    args = {"query": "retry"}

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return ToolMessage(
                content="temporary failure",
                name="search_docs",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return ToolMessage(
            content="success",
            name="search_docs",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first_request = _request("search_docs", "call-1", args)
        first = await asyncio.create_task(
            middleware.awrap_tool_call(first_request, handler)
        )
        second_request = _request(
            "search_docs",
            "call-2",
            args,
            _history("search_docs", "call-1", args, first),
        )
        second = await asyncio.create_task(
            middleware.awrap_tool_call(second_request, handler)
        )
        return second

    second = asyncio.run(invoke())

    assert len(calls) == 2
    assert second.content == "success"
