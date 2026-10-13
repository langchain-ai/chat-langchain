"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.duplicate_call_guard_middleware import (
    _CHECK_LINKS_REFUSAL,
    _DUPLICATE_NOTE,
    DuplicateCallGuardMiddleware,
)


def _request(name: str, call_id: str, args: dict, history=None):
    tool_call = {"name": name, "id": call_id, "args": args}
    if history is None:
        history = [HumanMessage(content="Question")]
    return ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state={
            "messages": [
                *history,
                AIMessage(content="", tool_calls=[tool_call]),
            ]
        },
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

    request = _request("search_docs", "call-1", {"query": "middleware"})
    first = asyncio.run(middleware.awrap_tool_call(request, handler))
    second = asyncio.run(
        middleware.awrap_tool_call(
            _request(
                "search_docs",
                "call-2",
                {"query": "middleware"},
                [*request.state["messages"], first],
            ),
            handler,
        )
    )

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
        request = _request("search_docs", "call-1", {"query": "first"})
        first = await middleware.awrap_tool_call(request, handler)
        await middleware.awrap_tool_call(
            _request(
                "search_docs",
                "call-2",
                {"query": "second"},
                [*request.state["messages"], first],
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
        request = _request("search_docs", "call-1", {"query": "retry"})
        with pytest.raises(RuntimeError):
            await middleware.awrap_tool_call(request, handler)
        return await middleware.awrap_tool_call(
            _request(
                "search_docs", "call-2", {"query": "retry"}, request.state["messages"]
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

    request = _request("check_links", "call-1", {"urls": ["https://one.example"]})
    first = asyncio.run(middleware.awrap_tool_call(request, handler))
    second = asyncio.run(
        middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-2",
                {"urls": ["https://two.example"]},
                [*request.state["messages"], first],
            ),
            handler,
        )
    )

    assert len(calls) == 1
    assert first.content == "validated"
    assert second.content == _CHECK_LINKS_REFUSAL
    assert second.name == "check_links"
    assert second.tool_call_id == "call-2"


@pytest.mark.parametrize("status", ["success", "error", None])
def test_check_links_budget_counts_attempts_not_results(status):
    middleware = DuplicateCallGuardMiddleware()
    request = _request("check_links", "call-1", {"urls": []})
    history = request.state["messages"]
    if status is not None:
        history = [
            *history,
            ToolMessage(content="result", tool_call_id="call-1", status=status),
        ]

    async def handler(request):
        pytest.fail("A second check_links attempt must not reach the handler")

    result = asyncio.run(
        middleware.awrap_tool_call(
            _request("check_links", "call-2", {"urls": []}, history), handler
        )
    )

    assert result.content == _CHECK_LINKS_REFUSAL


def test_check_links_batch_uses_call_order_even_when_handlers_start_in_reverse():
    middleware = DuplicateCallGuardMiddleware()
    first = _request("check_links", "call-1", {"urls": ["https://one.example"]})
    second = _request("check_links", "call-2", {"urls": ["https://two.example"]})
    first.state["messages"][-1].tool_calls.append(second.tool_call)
    second = second.override(state=first.state)
    calls = []

    async def handler(request):
        calls.append(request.tool_call["id"])
        await asyncio.sleep(0)
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        return await asyncio.gather(
            middleware.awrap_tool_call(second, handler),
            middleware.awrap_tool_call(first, handler),
        )

    refused, allowed = asyncio.run(invoke())

    assert calls == ["call-1"]
    assert refused.content == _CHECK_LINKS_REFUSAL
    assert refused.tool_call_id == "call-2"
    assert allowed.content == "validated"


@pytest.mark.parametrize("state", [None, {}, {"messages": []}, {"messages": None}])
def test_missing_history_falls_through(state):
    middleware = DuplicateCallGuardMiddleware()
    request = _request("check_links", "call-1", {"urls": []}).override(state=state)
    expected = ToolMessage(content="validated", tool_call_id="call-1")

    async def handler(request):
        return expected

    assert asyncio.run(middleware.awrap_tool_call(request, handler)) is expected


@pytest.mark.parametrize(
    ("prior_name", "result_id", "status", "cached"),
    [
        ("search_docs", "call-1", "success", True),
        ("search_docs", "call-1", "error", False),
        ("search_docs", "unrelated", "success", False),
        ("other_tool", "call-1", "success", False),
    ],
)
def test_duplicates_match_canonical_args_and_successful_result_ids(
    prior_name, result_id, status, cached
):
    middleware = DuplicateCallGuardMiddleware()
    prior = _request(
        prior_name, "call-1", {"query": "docs", "options": {"b": 2, "a": 1}}
    )
    content = [{"type": "text", "text": "cached result"}]
    history = [
        *prior.state["messages"],
        ToolMessage(content=content, tool_call_id=result_id, status=status),
    ]
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="fresh result", tool_call_id=request.tool_call["id"])

    result = asyncio.run(
        middleware.awrap_tool_call(
            _request(
                "search_docs",
                "call-2",
                {"options": {"a": 1, "b": 2}, "query": "docs"},
                history,
            ),
            handler,
        )
    )

    assert len(calls) == (0 if cached else 1)
    if cached:
        assert result.content == (
            _DUPLICATE_NOTE + '\n[{"type": "text", "text": "cached result"}]'
        )
        assert result.name == "search_docs"
        assert result.tool_call_id == "call-2"
    else:
        assert result.content == "fresh result"


class _ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("tool_name", ["check_links", "search_docs"])
def test_real_agent_suppresses_calls_across_supersteps_and_resets_on_new_human(
    tool_name,
):
    calls = []

    async def counting_tool(query: str) -> str:
        calls.append(query)
        return "cached result"

    model = _ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "id": f"call-{call_id}",
                        "args": {"query": "docs"},
                    }
                ],
            )
            if call_id not in (3, 6)
            else AIMessage(content="Answer")
            for call_id in range(1, 7)
        ]
    )
    graph = create_agent(
        model=model,
        tools=[
            StructuredTool.from_function(
                coroutine=counting_tool,
                name=tool_name,
                description="Count tool executions",
            )
        ],
        middleware=[DuplicateCallGuardMiddleware()],
    )
    first = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="Question")]}))

    assert calls == ["docs"]
    results = [
        message for message in first["messages"] if isinstance(message, ToolMessage)
    ]
    expected = (
        _CHECK_LINKS_REFUSAL
        if tool_name == "check_links"
        else f"{_DUPLICATE_NOTE}\ncached result"
    )
    assert len(results) == 2
    assert results[0].content == "cached result"
    assert results[1].content == expected
    assert results[1].name == tool_name
    assert results[1].tool_call_id == "call-2"
    assert first["messages"][-1].content == "Answer"

    second = asyncio.run(
        graph.ainvoke(
            {"messages": [*first["messages"], HumanMessage(content="Question")]}
        )
    )

    assert calls == ["docs", "docs"]
    results = [
        message for message in second["messages"] if isinstance(message, ToolMessage)
    ]
    assert results[-2].content == "cached result"
    assert results[-1].content == expected
    assert results[-1].tool_call_id == "call-5"
