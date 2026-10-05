"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
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
        second = await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "middleware"}, state=state),
            handler,
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
    state = {"messages": [HumanMessage(content="Question")]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content=request.tool_call["args"]["query"],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "first"}, state=state), handler
        )
        await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "second"}, state=state), handler
        )

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
        return await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "retry"}, state=state), handler
        )

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
    assert first.content == "validated"
    assert "may only be called once per turn" in second.content


def test_identical_check_links_is_suppressed_across_tasks_and_graph_steps():
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
        first = await asyncio.create_task(
            middleware.awrap_tool_call(
                _request(
                    "check_links",
                    "call-1",
                    {"urls": ["https://one.example"]},
                    state=state,
                ),
                handler,
            )
        )
        state["messages"].append(first)
        second_state = {**state, "messages": list(state["messages"])}
        second = await asyncio.create_task(
            middleware.awrap_tool_call(
                _request(
                    "check_links",
                    "call-2",
                    {"urls": ["https://one.example"]},
                    state=second_state,
                ),
                handler,
            )
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert second.tool_call_id == "call-2"
    assert "already made on this turn" in second.content
    assert "validated" in second.content


def test_new_human_message_resets_shared_turn_state():
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    state = {"messages": [HumanMessage(content="First question")]}

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "same"}, state=state),
            handler,
        )
        state["messages"].append(HumanMessage(content="Second question"))
        return await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "same"}, state=state),
            handler,
        )

    result = asyncio.run(invoke())

    assert len(calls) == 2
    assert result.content == "result"


def test_repeated_call_ceiling_terminates_with_answer_and_links():
    middleware = DuplicateCallGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Question"),
            AIMessage(content="Here is the answer."),
            ToolMessage(
                content="https://one.example is valid",
                name="check_links",
                tool_call_id="link-call",
            ),
        ]
    }

    async def handler(request):
        return ToolMessage(
            content="cached result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        results = []
        for index in range(4):
            results.append(
                await middleware.awrap_tool_call(
                    _request(
                        "search_docs",
                        f"call-{index}",
                        {"query": "same"},
                        state=state,
                    ),
                    handler,
                )
            )
        return results

    results = asyncio.run(invoke())

    assert all(isinstance(result, (ToolMessage, Command)) for result in results)
    assert isinstance(results[-1], Command)
    assert results[-1].goto == END
    final_message = results[-1].update["messages"][0]
    assert final_message.content.startswith("Here is the answer.")
    assert "https://one.example is valid" in final_message.content
