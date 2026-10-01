"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware


def _request(
    name: str,
    call_id: str,
    args: dict,
    content: str = "Question",
    messages: list | None = None,
):
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state={"messages": messages or [HumanMessage(content=content)]},
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
        messages = [HumanMessage(content="Question")]
        first = await middleware.awrap_tool_call(
            _request(
                "search_docs", "call-1", {"query": "middleware"}, messages=messages
            ),
            handler,
        )
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_docs",
                            "args": {"query": "middleware"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                first,
            ]
        )
        second = await middleware.awrap_tool_call(
            _request(
                "search_docs", "call-2", {"query": "middleware"}, messages=messages
            ),
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

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content=request.tool_call["args"]["query"],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        messages = [HumanMessage(content="Question")]
        await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "first"}, messages=messages),
            handler,
        )
        await middleware.awrap_tool_call(
            _request(
                "search_docs", "call-2", {"query": "second"}, messages=messages
            ),
            handler,
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
        messages = [HumanMessage(content="Question")]
        request = _request(
            "search_docs", "call-1", {"query": "retry"}, messages=messages
        )
        with pytest.raises(RuntimeError):
            await middleware.awrap_tool_call(request, handler)
        return await middleware.awrap_tool_call(
            _request(
                "search_docs", "call-2", {"query": "retry"}, messages=messages
            ),
            handler,
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
        messages = [HumanMessage(content="Question")]
        first = await middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-1",
                {"urls": ["https://one.example"]},
                messages=messages,
            ),
            handler,
        )
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_links",
                            "args": {"urls": ["https://one.example"]},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                first,
            ]
        )
        second = await middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-2",
                {"urls": ["https://two.example"]},
                messages=messages,
            ),
            handler,
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert "may only be called once per turn" in second.content


def test_duplicate_call_suppression_uses_history_across_tasks():
    middleware = DuplicateCallGuardMiddleware()
    messages = [HumanMessage(content="Question")]
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="cached result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke_first():
        return await middleware.awrap_tool_call(
            _request("search_docs", "call-1", {"query": "middleware"}, messages=messages),
            handler,
        )

    first = asyncio.run(invoke_first())
    messages.extend(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_docs",
                        "args": {"query": "middleware"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            first,
        ]
    )

    async def invoke_second():
        return await middleware.awrap_tool_call(
            _request("search_docs", "call-2", {"query": "middleware"}, messages=messages),
            handler,
        )

    second = asyncio.run(invoke_second())

    assert len(calls) == 1
    assert second.tool_call_id == "call-2"
    assert "already made on this turn" in second.content


def test_research_guard_uses_persisted_retry_history_across_tasks():
    middleware = DocsResearchGuardMiddleware()
    messages = [HumanMessage(content="How do I configure StateGraph?")]
    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="Use the StateGraph config_schema parameter for this setup."
                )
            ]
        )

    async def invoke_first():
        return await middleware.awrap_model_call(
            ModelRequest(model=object(), messages=messages), handler
        )

    first = asyncio.run(invoke_first())
    messages[:] = [*calls[-1].messages, *first.result]
    call_count = len(calls)

    async def invoke_second():
        return await middleware.awrap_model_call(
            ModelRequest(model=object(), messages=messages), handler
        )

    asyncio.run(invoke_second())

    assert len(calls) == call_count + 2
