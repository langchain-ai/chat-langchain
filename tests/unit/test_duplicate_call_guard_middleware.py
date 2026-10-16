"""Tests for per-turn duplicate tool-call suppression."""

import asyncio
import contextvars

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolNode, ToolRuntime

from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware


def _request(
    name: str,
    call_id: str,
    args: dict,
    content: str = "Question",
    thread_id: str = "thread-1",
    human_id: str | None = "human-1",
):
    state = {"messages": [HumanMessage(content=content, id=human_id)]}
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


async def _invoke_in_task(middleware, request, handler):
    return await asyncio.create_task(
        middleware.awrap_tool_call(request, handler), context=contextvars.copy_context()
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
        first = await _invoke_in_task(
            middleware,
            _request("search_docs", "call-1", {"query": "middleware"}),
            handler,
        )
        second = await _invoke_in_task(
            middleware,
            _request("search_docs", "call-2", {"query": "middleware"}),
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
            await _invoke_in_task(middleware, request, handler)
        return await _invoke_in_task(
            middleware, _request("search_docs", "call-2", {"query": "retry"}), handler
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
        first = await _invoke_in_task(
            middleware,
            _request("check_links", "call-1", {"urls": ["https://one.example"]}),
            handler,
        )
        second = await _invoke_in_task(
            middleware,
            _request("check_links", "call-2", {"urls": ["https://two.example"]}),
            handler,
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert "may only be called once per turn" in second.content


def test_error_tool_message_is_not_cached():
    middleware = DuplicateCallGuardMiddleware()
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        return ToolMessage(
            content="failed" if attempts == 1 else "success",
            status="error" if attempts == 1 else "success",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first = await _invoke_in_task(
            middleware, _request("search_docs", "call-1", {}), handler
        )
        second = await _invoke_in_task(
            middleware, _request("search_docs", "call-2", {}), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert attempts == 2
    assert first.status == "error"
    assert second.content == "success"


@pytest.mark.parametrize("name", ["check_links", "search_docs"])
def test_new_turns_and_threads_do_not_share_state(name):
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="fresh result", tool_call_id=request.tool_call["id"])

    async def invoke():
        requests = [
            _request(name, "call-1", {}),
            _request(name, "call-2", {}, thread_id="thread-2"),
            _request(name, "call-3", {}, human_id="human-2"),
        ]
        results = [
            await _invoke_in_task(middleware, request, handler) for request in requests
        ]
        repeated = await _invoke_in_task(
            middleware, _request(name, "call-4", {}, thread_id="thread-2"), handler
        )
        return results, repeated

    results, repeated = asyncio.run(invoke())

    assert len(calls) == 3
    assert all(result.content == "fresh result" for result in results)
    assert repeated.content != "fresh result"
    assert len(middleware._turn_states) == 2


def test_parallel_check_links_calls_share_one_budget():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0)
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        return await asyncio.gather(
            *(
                _invoke_in_task(
                    middleware, _request("check_links", f"call-{index}", {}), handler
                )
                for index in range(3)
            )
        )

    results = asyncio.run(invoke())

    assert len(calls) == 1
    assert sum(result.content == "validated" for result in results) == 1
    assert (
        sum("may only be called once per turn" in result.content for result in results)
        == 2
    )


def test_thread_key_is_stable_when_step_run_ids_change():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        for index in range(2):
            request = _request("check_links", f"call-{index}", {})
            request.runtime.config["run_id"] = f"run-{index}"
            result = await _invoke_in_task(middleware, request, handler)
        return result

    result = asyncio.run(invoke())

    assert len(calls) == 1
    assert "may only be called once per turn" in result.content


def test_turn_identity_survives_message_history_trimming():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        first = _request("check_links", "call-1", {})
        first.state["messages"].insert(
            0, HumanMessage(content="Earlier question", id="old-human")
        )
        await _invoke_in_task(middleware, first, handler)
        return await _invoke_in_task(
            middleware, _request("check_links", "call-2", {}), handler
        )

    result = asyncio.run(invoke())

    assert len(calls) == 1
    assert "may only be called once per turn" in result.content


def test_idless_turn_uses_content_and_position():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        first = _request("check_links", "call-1", {}, human_id=None)
        await _invoke_in_task(middleware, first, handler)
        repeated = await _invoke_in_task(
            middleware, _request("check_links", "call-2", {}, human_id=None), handler
        )
        next_turn = _request("check_links", "call-3", {}, human_id=None)
        next_turn.state["messages"].insert(0, first.state["messages"][0])
        fresh = await _invoke_in_task(middleware, next_turn, handler)
        return repeated, fresh

    repeated, fresh = asyncio.run(invoke())

    assert len(calls) == 2
    assert "may only be called once per turn" in repeated.content
    assert fresh.content == "validated"


def test_turn_state_mapping_evicts_least_recently_used_threads(monkeypatch):
    monkeypatch.setattr(
        "src.middleware.duplicate_call_guard_middleware._MAX_TURN_STATES", 2
    )
    middleware = DuplicateCallGuardMiddleware()

    async def handler(request):
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        for thread_id in ("thread-1", "thread-2", "thread-1", "thread-3"):
            await _invoke_in_task(
                middleware,
                _request("check_links", thread_id, {}, thread_id=thread_id),
                handler,
            )

    asyncio.run(invoke())

    assert list(middleware._turn_states) == ["thread:thread-1", "thread:thread-3"]


@pytest.mark.parametrize("name", ["check_links", "search_docs"])
@pytest.mark.parametrize("with_thread", [False, True])
def test_successive_graph_tool_steps_share_state_and_reset_on_new_turn(
    name, with_thread
):
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    @tool(name)
    async def guarded_tool(query: str) -> str:
        """Return a fixed tool result."""
        calls.append(query)
        return "tool result"

    async def model(state):
        messages = state["messages"]
        latest_human = max(
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        )
        attempts = sum(
            isinstance(message, ToolMessage) for message in messages[latest_human + 1 :]
        )
        if attempts == 2:
            return {"messages": [AIMessage(content="done")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": {"query": "same query"},
                            "id": f"{messages[latest_human].id}-{attempts}",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_node(
        "tools", ToolNode([guarded_tool], awrap_tool_call=middleware.awrap_tool_call)
    )
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", lambda state: "tools" if state["messages"][-1].tool_calls else END
    )
    builder.add_edge("tools", "model")
    graph = builder.compile(checkpointer=InMemorySaver() if with_thread else None)

    async def invoke():
        config = (
            {"configurable": {"thread_id": "graph-thread"}} if with_thread else None
        )
        first = await graph.ainvoke(
            {"messages": [HumanMessage(content="Question")]}, config
        )
        second = await graph.ainvoke(
            {"messages": [HumanMessage(content="Question")]}, config
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 2
    for result in (first, second):
        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        assert tool_messages[-2].content == "tool result"
        assert tool_messages[-1].tool_call_id != tool_messages[-2].tool_call_id
        expected = (
            "may only be called once per turn"
            if name == "check_links"
            else "already made on this turn"
        )
        assert expected in tool_messages[-1].content
