"""Tests for per-turn duplicate tool-call suppression."""

import asyncio
from contextvars import copy_context
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime
from langgraph.runtime import Runtime

from src.middleware import duplicate_call_guard_middleware as guard
from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware


@pytest.fixture(autouse=True)
def clear_turn_state():
    guard._TURN_STATE.clear()
    yield
    guard._TURN_STATE.clear()


def _request(
    name: str,
    call_id: str,
    args: dict,
    content: str = "Question",
    thread_id: str | None = "test-thread",
    human_id: str | None = None,
    invocation_id: str | None = None,
):
    state = {
        "messages": [HumanMessage(content=content, id=human_id)],
        "duplicate_call_guard_invocation": invocation_id,
    }
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state=state,
        runtime=ToolRuntime(
            state=state,
            context=None,
            config={
                "configurable": {"thread_id": thread_id} if thread_id else {},
                "run_id": uuid4(),
            },
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


@pytest.mark.parametrize("tool_name", ["search_docs", "check_links"])
def test_identical_calls_share_state_across_tasks_with_copied_context(tool_name):
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    human = HumanMessage(content="Question", id="human-turn")

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="result", tool_call_id=request.tool_call["id"])

    async def invoke():
        results = []
        for call_id in ["call-1", "call-2"]:
            request = _request(tool_name, call_id, {"query": "same"})
            request.state["messages"] = [human]
            results.append(
                await asyncio.create_task(
                    middleware.awrap_tool_call(request, handler), context=copy_context()
                )
            )
        return results

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "result"
    assert second.tool_call_id == "call-2"
    if tool_name == "check_links":
        assert second.content == guard._CHECK_LINKS_REFUSAL
    else:
        assert "already made on this turn" in second.content
        assert "result" in second.content


def test_parallel_check_links_reserves_budget_before_execution_finishes():
    middleware = DuplicateCallGuardMiddleware()
    calls = []

    async def invoke():
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(request):
            calls.append(request)
            started.set()
            await release.wait()
            return ToolMessage(
                content="validated", tool_call_id=request.tool_call["id"]
            )

        first = asyncio.create_task(
            middleware.awrap_tool_call(
                _request("check_links", "call-1", {"urls": ["https://one.example"]}),
                handler,
            ),
            context=copy_context(),
        )
        await started.wait()
        second = await asyncio.create_task(
            middleware.awrap_tool_call(
                _request("check_links", "call-2", {"urls": ["https://two.example"]}),
                handler,
            ),
            context=copy_context(),
        )
        release.set()
        return await first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "validated"
    assert second.content == guard._CHECK_LINKS_REFUSAL


@pytest.mark.parametrize("thread_id", ["test-thread", None])
@pytest.mark.parametrize("tool_schema", [False, True])
def test_model_removes_check_links_only_after_refusal(
    monkeypatch, thread_id, tool_schema
):
    middleware = DuplicateCallGuardMiddleware()
    requests = []

    @tool
    def check_links(urls: list[str]) -> str:
        """Validate links."""
        return "validated"

    link_tool = {"name": "check_links"} if tool_schema else check_links
    tools = [link_tool, {"name": "search_docs"}]
    first_request = _request(
        "check_links",
        "call-1",
        {"urls": []},
        thread_id=thread_id,
        invocation_id="scope",
    )
    monkeypatch.setattr(guard, "get_config", lambda: first_request.runtime.config)
    model_request = ModelRequest(
        model=object(),
        messages=first_request.state["messages"],
        state=first_request.state,
        tools=tools,
        runtime=Runtime(),
    )

    async def model_handler(request):
        requests.append(request)
        return ModelResponse(result=[AIMessage(content="answer")])

    async def tool_handler(request):
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        await middleware.awrap_model_call(model_request, model_handler)
        await middleware.awrap_tool_call(first_request, tool_handler)
        await middleware.awrap_model_call(model_request, model_handler)
        refused = await middleware.awrap_tool_call(
            _request(
                "check_links",
                "call-2",
                {"urls": []},
                thread_id=thread_id,
                invocation_id="scope",
            ),
            tool_handler,
        )
        await middleware.awrap_model_call(model_request, model_handler)
        return refused

    refused = asyncio.run(invoke())

    assert refused.content == guard._CHECK_LINKS_REFUSAL
    assert requests[0].tools == tools
    assert requests[1].tools == tools
    assert requests[2].tools == [{"name": "search_docs"}]
    assert model_request.tools == tools


@pytest.mark.parametrize("scope", ["new-thread", "new-turn", "new-invocation"])
def test_unrelated_turns_and_invocations_do_not_share_budget(scope):
    middleware = DuplicateCallGuardMiddleware()
    calls = []
    first = _request("check_links", "call-1", {}, human_id="human-1")
    second = _request("check_links", "call-2", {}, human_id="human-1")
    if scope == "new-thread":
        second.runtime.config["configurable"]["thread_id"] = "other-thread"
    elif scope == "new-turn":
        second.state["messages"] = [HumanMessage(content="Question", id="human-2")]
    else:
        for request in [first, second]:
            request.runtime.config["configurable"] = {}
            request.state.update(middleware.before_agent(request.state, Runtime()))

    async def handler(request):
        calls.append(request)
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        await middleware.awrap_tool_call(first, handler)
        return await middleware.awrap_tool_call(second, handler)

    second_result = asyncio.run(invoke())

    assert len(calls) == 2
    assert second_result.content == "validated"


def test_turn_state_is_bounded_and_recent_entries_are_retained(monkeypatch):
    middleware = DuplicateCallGuardMiddleware()
    monkeypatch.setattr(guard, "_MAX_TURN_STATES", 2)

    async def handler(request):
        return ToolMessage(content="validated", tool_call_id=request.tool_call["id"])

    async def invoke():
        for thread_id in ["old", "recent", "old", "new"]:
            await middleware.awrap_tool_call(
                _request("check_links", thread_id, {}, thread_id=thread_id), handler
            )

    asyncio.run(invoke())

    assert len(guard._TURN_STATE) == 2
    assert {key[0] for key in guard._TURN_STATE} == {"thread:old", "thread:new"}


class _ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _RecordToolsMiddleware(AgentMiddleware):
    def __init__(self):
        self.available_tools = []

    async def awrap_model_call(self, request, handler):
        self.available_tools.append([tool.name for tool in request.tools])
        return await handler(request)


@pytest.mark.parametrize("thread_id", ["compiled-thread", None])
def test_compiled_agent_refuses_cross_step_duplicate_and_reaches_final_answer(
    thread_id,
):
    executions = []

    @tool
    async def check_links(urls: list[str]) -> str:
        """Validate links."""
        executions.append(urls)
        return "Link Check Results: 1/1 valid"

    model = _ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_links",
                        "id": call_id,
                        "args": {"urls": ["https://docs.langchain.com"]},
                    }
                ],
            )
            for call_id in ["call-1", "call-2"]
        ]
        + [AIMessage(content="Final answer using the validated link.")]
    )
    recorder = _RecordToolsMiddleware()
    agent = create_agent(
        model=model,
        tools=[check_links],
        middleware=[DuplicateCallGuardMiddleware(), recorder],
    )
    config = {"recursion_limit": 12}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    async def invoke():
        results = []
        for _ in range(2):
            results.append(
                await agent.ainvoke(
                    {"messages": [HumanMessage(content="Question")]}, config
                )
            )
        return results

    results = asyncio.run(invoke())

    assert len(executions) == 2
    assert recorder.available_tools == [["check_links"], ["check_links"], []] * 2
    for result in results:
        messages = result["messages"]
        assert messages[-1].content == "Final answer using the validated link."
        tool_results = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        assert len(tool_results) == 2
        assert tool_results[0].content == "Link Check Results: 1/1 valid"
        assert tool_results[1].content == guard._CHECK_LINKS_REFUSAL
        assert tool_results[1].tool_call_id == "call-2"
        assert "duplicate_call_guard_invocation" not in result
