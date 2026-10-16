"""Tests for per-turn duplicate tool-call suppression."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime

from src.middleware.duplicate_call_guard_middleware import (
    _CHECK_LINKS_REFUSAL,
    _DUPLICATE_NOTE,
    _MAX_CACHED_CALLS,
    _MAX_TURN_STATES,
    DuplicateCallGuardMiddleware,
)


def _request(
    name: str,
    call_id: str,
    args: dict,
    content: str = "Question",
    messages=None,
    thread_id="thread-1",
):
    state = {
        "messages": messages
        if messages is not None
        else [HumanMessage(content=content, id="human-1")]
    }
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state=state,
        runtime=ToolRuntime(
            state=state,
            context=None,
            config={"configurable": {"thread_id": thread_id}},
            stream_writer=lambda chunk: None,
            tool_call_id=call_id,
            store=None,
        ),
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


def _history(name, args, content="cached result", status="success"):
    return [
        HumanMessage(content="Question", id="human-1"),
        AIMessage(
            content="",
            id="ai-1",
            tool_calls=[{"name": name, "id": "call-1", "args": args}],
        ),
        ToolMessage(content=content, tool_call_id="call-1", status=status),
    ]


@pytest.mark.parametrize(
    "content", ["cached result", [{"type": "text", "text": "result"}]]
)
def test_history_caches_success_across_separate_tasks(content):
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    messages = [HumanMessage(content="Question", id="human-1")]
    args = {"query": "middleware", "options": {"limit": 5, "language": "python"}}

    async def handler(request):
        calls.append(request)
        return ToolMessage(content=content, tool_call_id=request.tool_call["id"])

    async def invoke():
        first = _request("search_docs", "call-1", args, messages=messages)
        messages.append(AIMessage(content="", tool_calls=[first.tool_call]))
        result = await asyncio.create_task(middleware.awrap_tool_call(first, handler))
        messages.append(result)
        second = _request(
            "search_docs",
            "call-2",
            {"options": {"language": "python", "limit": 5}, "query": "middleware"},
            messages=messages,
        )
        messages.append(AIMessage(content="", tool_calls=[second.tool_call]))
        return await asyncio.create_task(
            DuplicateCallGuardMiddleware().awrap_tool_call(second, handler)
        )

    result = asyncio.run(invoke())

    assert len(calls) == 1
    assert result.name == "search_docs"
    assert result.tool_call_id == "call-2"
    assert result.content == f"{_DUPLICATE_NOTE}\n{middleware._content_text(content)}"


@pytest.mark.parametrize("status", ["success", "error", None])
@pytest.mark.parametrize("args", [{"urls": ["one"]}, {"urls": ["two"]}])
def test_history_consumes_check_links_budget_even_without_success(status, args):
    middleware = DuplicateCallGuardMiddleware()
    messages = _history("check_links", {"urls": ["one"]}, status=status or "error")
    if status is None:
        messages.pop()
    request = _request("check_links", "call-2", args, messages=messages)
    messages.append(AIMessage(content="", tool_calls=[request.tool_call]))

    async def handler(request):
        pytest.fail("An earlier invocation must consume the check_links budget")

    async def invoke():
        return await asyncio.create_task(middleware.awrap_tool_call(request, handler))

    result = asyncio.run(invoke())

    assert result.content == _CHECK_LINKS_REFUSAL
    assert result.name == "check_links"
    assert result.tool_call_id == "call-2"


@pytest.mark.parametrize(
    ("name", "args", "status"),
    [
        ("other_tool", {"query": "middleware"}, "success"),
        ("search_docs", {"query": "different"}, "success"),
        ("search_docs", {"query": "middleware"}, "error"),
    ],
)
def test_history_does_not_suppress_different_or_failed_calls(name, args, status):
    middleware = DuplicateCallGuardMiddleware()
    messages = _history("search_docs", {"query": "middleware"}, status=status)
    request = _request(name, "call-2", args, messages=messages)
    messages.append(AIMessage(content="", tool_calls=[request.tool_call]))

    async def handler(request):
        return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

    async def invoke():
        return await asyncio.create_task(middleware.awrap_tool_call(request, handler))

    assert asyncio.run(invoke()).content == "executed"


@pytest.mark.parametrize("failure", ["error", "exception"])
def test_check_links_reservation_survives_unsuccessful_invocation(failure):
    middleware = DuplicateCallGuardMiddleware()
    messages = [HumanMessage(content="Question", id="human-1")]
    first = _request("check_links", "call-1", {"urls": ["one"]}, messages=messages)
    messages.append(AIMessage(content="", tool_calls=[first.tool_call]))
    calls = []

    async def handler(request):
        calls.append(request)
        if failure == "exception":
            raise RuntimeError("temporary failure")
        return ToolMessage(content="failed", tool_call_id="call-1", status="error")

    async def invoke():
        if failure == "exception":
            with pytest.raises(RuntimeError):
                await asyncio.create_task(middleware.awrap_tool_call(first, handler))
        else:
            await asyncio.create_task(middleware.awrap_tool_call(first, handler))
        second = _request("check_links", "call-2", {"urls": ["two"]}, messages=messages)
        messages.append(AIMessage(content="", tool_calls=[second.tool_call]))
        return await asyncio.create_task(middleware.awrap_tool_call(second, handler))

    assert asyncio.run(invoke()).content == _CHECK_LINKS_REFUSAL
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["search_docs", "check_links"])
@pytest.mark.parametrize("missing_runtime", [False, True])
def test_parallel_pending_calls_share_one_successful_execution(name, missing_runtime):
    middleware = DuplicateCallGuardMiddleware()
    messages = [HumanMessage(content="Question", id="human-1")]
    first = _request(name, "call-1", {"query": "same"}, messages=messages)
    second = _request(name, "call-2", {"query": "same"}, messages=messages)
    if name == "check_links":
        second.tool_call["args"] = {"query": "different"}
    if missing_runtime:
        first = first.override(runtime=None)
        second = second.override(runtime=None)
    messages.append(
        AIMessage(content="", id="ai-1", tool_calls=[first.tool_call, second.tool_call])
    )
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0)
        return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

    async def invoke():
        return await asyncio.gather(
            asyncio.create_task(middleware.awrap_tool_call(first, handler)),
            asyncio.create_task(middleware.awrap_tool_call(second, handler)),
        )

    results = asyncio.run(invoke())

    assert len(calls) == 1
    assert results[0].content == "executed"
    expected = (
        _CHECK_LINKS_REFUSAL
        if name == "check_links"
        else f"{_DUPLICATE_NOTE}\nexecuted"
    )
    assert results[1].content == expected
    assert results[1].tool_call_id == "call-2"
    assert results[1].name == name


@pytest.mark.parametrize("failure", ["error", "exception", "command"])
def test_parallel_unsuccessful_non_budgeted_calls_remain_retryable(failure):
    from langgraph.types import Command

    middleware = DuplicateCallGuardMiddleware()
    messages = [HumanMessage(content="Question", id="human-1")]
    requests = [
        _request("search_docs", f"call-{index}", {"query": "same"}, messages=messages)
        for index in range(3)
    ]
    messages.append(
        AIMessage(content="", tool_calls=[request.tool_call for request in requests])
    )
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0)
        if len(calls) == 1:
            if failure == "exception":
                raise RuntimeError("temporary failure")
            if failure == "command":
                return Command(update={})
            return ToolMessage(
                content="failed", tool_call_id=request.tool_call["id"], status="error"
            )
        return ToolMessage(content="success", tool_call_id=request.tool_call["id"])

    async def invoke():
        return await asyncio.gather(
            *(
                asyncio.create_task(middleware.awrap_tool_call(request, handler))
                for request in requests
            ),
            return_exceptions=True,
        )

    results = asyncio.run(invoke())

    assert len(calls) == 2
    assert results[1].content == "success"
    assert results[2].content == f"{_DUPLICATE_NOTE}\nsuccess"


@pytest.mark.parametrize("name", ["check_links", "search_docs"])
@pytest.mark.parametrize("new_scope", ["thread", "turn", "missing_runtime"])
def test_other_threads_and_new_human_turns_are_not_suppressed(name, new_scope):
    middleware = DuplicateCallGuardMiddleware()
    messages = [HumanMessage(content="Question", id="human-1")]
    first = _request(name, "call-1", {}, messages=messages)
    messages.append(AIMessage(content="", tool_calls=[first.tool_call]))
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

    async def invoke():
        if new_scope == "missing_runtime":
            first_request = first.override(runtime=None)
        else:
            first_request = first
        messages.append(
            await asyncio.create_task(
                middleware.awrap_tool_call(first_request, handler)
            )
        )
        second_messages = messages.copy()
        if new_scope == "turn":
            second_messages.append(HumanMessage(content="Question", id="human-2"))
        elif new_scope == "thread":
            second_messages = [messages[0]]
        else:
            second_messages = [HumanMessage(content="Question", id="human-1")]
        second = _request(
            name,
            "call-2",
            {},
            messages=second_messages,
            thread_id="thread-2" if new_scope == "thread" else "thread-1",
        )
        if new_scope == "missing_runtime":
            second = second.override(runtime=None)
        second_messages.append(AIMessage(content="", tool_calls=[second.tool_call]))
        return await asyncio.create_task(middleware.awrap_tool_call(second, handler))

    assert asyncio.run(invoke()).content == "executed"
    assert len(calls) == 2


def test_coordination_storage_is_bounded():
    middleware = DuplicateCallGuardMiddleware()

    async def handler(request):
        return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

    async def invoke():
        for index in range(_MAX_TURN_STATES + 10):
            await middleware.awrap_tool_call(
                _request("search_docs", "call-1", {}, thread_id=f"thread-{index}"),
                handler,
            )
        assert len(middleware._turn_states) == _MAX_TURN_STATES
        for index in range(_MAX_CACHED_CALLS + 10):
            await middleware.awrap_tool_call(
                _request("search_docs", f"call-{index}", {"query": index}), handler
            )
        assert (
            len(next(reversed(middleware._turn_states.values())).seen_calls)
            == _MAX_CACHED_CALLS
        )

    asyncio.run(invoke())


def test_same_text_new_human_without_id_resets_budget():
    middleware = DuplicateCallGuardMiddleware()

    async def handler(request):
        return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

    async def invoke():
        for index in range(2):
            request = _request(
                "check_links",
                f"call-{index}",
                {},
                messages=[HumanMessage(content="Question")],
            )
            result = await asyncio.create_task(
                middleware.awrap_tool_call(request, handler)
            )
            assert result.content == "executed"

    asyncio.run(invoke())


def test_cancelled_waiter_does_not_cancel_shared_execution():
    middleware = DuplicateCallGuardMiddleware()
    messages = [HumanMessage(content="Question", id="human-1")]
    requests = [
        _request("search_docs", f"call-{index}", {}, messages=messages)
        for index in range(3)
    ]
    messages.append(
        AIMessage(content="", tool_calls=[request.tool_call for request in requests])
    )
    calls = []

    async def invoke():
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(request):
            calls.append(request)
            started.set()
            await release.wait()
            return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

        producer = asyncio.create_task(middleware.awrap_tool_call(requests[0], handler))
        await started.wait()
        waiter = asyncio.create_task(middleware.awrap_tool_call(requests[1], handler))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        remaining = asyncio.create_task(
            middleware.awrap_tool_call(requests[2], handler)
        )
        release.set()
        await producer
        return await remaining

    assert asyncio.run(invoke()).content == f"{_DUPLICATE_NOTE}\nexecuted"
    assert len(calls) == 1


def test_missing_and_malformed_history_is_ignored():
    middleware = DuplicateCallGuardMiddleware()
    messages = [
        HumanMessage(content="Question", id="human-1"),
        AIMessage.model_construct(content="", tool_calls=[None, {}, {"id": []}]),
        ToolMessage(content="orphaned", tool_call_id="unmatched"),
    ]

    async def handler(request):
        return ToolMessage(content="executed", tool_call_id=request.tool_call["id"])

    result = asyncio.run(
        middleware.awrap_tool_call(
            _request("search_docs", "call-1", {}, messages=messages), handler
        )
    )

    assert result.content == "executed"


def test_compiled_graph_suppresses_parallel_and_later_tool_node_calls():
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    middleware = DuplicateCallGuardMiddleware()
    calls = {"search_docs": 0, "check_links": 0}

    @tool
    async def search_docs(query: str) -> str:
        """Search documentation."""
        calls["search_docs"] += 1
        await asyncio.sleep(0)
        return "documentation"

    @tool
    async def check_links(urls: list[str]) -> str:
        """Validate links."""
        calls["check_links"] += 1
        await asyncio.sleep(0)
        return "validated"

    def tool_calls(step):
        return [
            {
                "name": "search_docs",
                "id": f"search-{step}",
                "args": {"query": "middleware"},
            },
            {
                "name": "check_links",
                "id": f"links-{step}",
                "args": {"urls": [f"https://{step}.example"]},
            },
        ]

    def repeat_calls(state):
        return {"messages": [AIMessage(content="", tool_calls=tool_calls(3))]}

    node = ToolNode(
        [search_docs, check_links], awrap_tool_call=middleware.awrap_tool_call
    )
    builder = StateGraph(MessagesState)
    builder.add_node("first_tools", node)
    builder.add_node("repeat_calls", repeat_calls)
    builder.add_node("later_tools", node)
    builder.add_edge(START, "first_tools")
    builder.add_edge("first_tools", "repeat_calls")
    builder.add_edge("repeat_calls", "later_tools")
    builder.add_edge("later_tools", END)

    async def invoke():
        return await builder.compile().ainvoke(
            {
                "messages": [
                    HumanMessage(content="Question"),
                    AIMessage(content="", tool_calls=tool_calls(1) + tool_calls(2)),
                ]
            },
            {"configurable": {"thread_id": "compiled-thread"}},
        )

    result = asyncio.run(invoke())
    results = {
        message.tool_call_id: message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }

    assert calls == {"search_docs": 1, "check_links": 1}
    assert results["search-2"].content == f"{_DUPLICATE_NOTE}\ndocumentation"
    assert results["links-2"].content == _CHECK_LINKS_REFUSAL
    assert results["search-3"].content == f"{_DUPLICATE_NOTE}\ndocumentation"
    assert results["links-3"].content == _CHECK_LINKS_REFUSAL
