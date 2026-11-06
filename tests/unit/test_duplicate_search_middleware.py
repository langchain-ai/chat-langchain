"""Behavioral coverage for successful-search deduplication."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.middleware.duplicate_search_middleware import (
    SEARCH_TOOLS,
    DuplicateSearchMiddleware,
)
from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def request(state, name, args=None, call_id="first"):
    return SimpleNamespace(
        state=state,
        tool_call={"name": name, "args": args or {"query": "docs"}, "id": call_id},
    )


def apply_result(state, result):
    if isinstance(result, Command):
        state.setdefault("successful_searches", {}).update(
            result.update["successful_searches"]
        )
        return result.update["messages"][0]
    return result


@pytest.mark.parametrize("name", sorted(SEARCH_TOOLS))
@pytest.mark.parametrize("asynchronous", [False, True])
def test_successful_repeat_and_different_arguments(name, asynchronous):
    middleware = DuplicateSearchMiddleware()
    state = {}
    executions = []

    def handler(req):
        executions.append(req.tool_call["args"])
        return ToolMessage(content="Search results", tool_call_id=req.tool_call["id"])

    async def ahandler(req):
        return handler(req)

    def invoke(req):
        if asynchronous:
            result = asyncio.run(middleware.awrap_tool_call(req, ahandler))
        else:
            result = middleware.wrap_tool_call(req, handler)
        return apply_result(state, result)

    assert invoke(request(state, name)).content == "Search results"
    state["messages"] = [AIMessage(content="Summarized conversation")]
    repeated = invoke(request(state, name, call_id="repeat"))
    assert repeated.tool_call_id == "repeat"
    assert "earlier result" in repeated.content
    invoke(request(state, name, {"query": "different"}))
    assert executions == [{"query": "docs"}, {"query": "different"}]

    state["messages"] = [HumanMessage(content="Search again")]
    reset = middleware.before_agent(state, None)
    state["successful_searches"] = reset["successful_searches"].value
    assert invoke(request(state, name)).content == "Search results"
    assert len(executions) == 3


@pytest.mark.parametrize(
    "failure",
    [
        ToolMessage(content="Failed", status="error", tool_call_id="first"),
        ToolMessage(content='{"error": "API key missing"}', tool_call_id="first"),
        ToolMessage(content='{"error": "Tool unavailable"}', tool_call_id="first"),
        ToolMessage(
            content=[{"type": "text", "text": '{"error": "timeout"}'}],
            tool_call_id="first",
        ),
        ToolMessage(content="Failed", artifact={"isError": True}, tool_call_id="first"),
        TimeoutError("timed out"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_failures_leave_search_eligible(failure, asynchronous):
    middleware = DuplicateSearchMiddleware()
    state = {}
    executions = []

    def handler(req):
        executions.append(req.tool_call)
        if len(executions) == 1:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return ToolMessage(content="Recovered", tool_call_id=req.tool_call["id"])

    async def ahandler(req):
        return handler(req)

    def invoke():
        req = request(state, "search_support_articles")
        result = (
            asyncio.run(middleware.awrap_tool_call(req, ahandler))
            if asynchronous
            else middleware.wrap_tool_call(req, handler)
        )
        return apply_result(state, result)

    if isinstance(failure, Exception):
        with pytest.raises(TimeoutError):
            invoke()
    else:
        invoke()
    assert invoke().content == "Recovered"
    assert "earlier result" in invoke().content
    assert len(executions) == 2


def test_argument_order_and_tool_names():
    middleware = DuplicateSearchMiddleware()
    state = {}

    def handler(req):
        return ToolMessage(content="Results", tool_call_id=req.tool_call["id"])

    first = request(state, "search_support_articles", {"a": 1, "b": 2})
    apply_result(state, middleware.wrap_tool_call(first, handler))
    reordered = request(state, "search_support_articles", {"b": 2, "a": 1})
    assert "earlier result" in middleware.wrap_tool_call(reordered, handler).content
    other = request(state, "langchainDocs_search_docs_by_lang_chain", {"b": 2, "a": 1})
    assert (
        apply_result(state, middleware.wrap_tool_call(other, handler)).content
        == "Results"
    )
    unrelated = request(state, "check_links")
    assert middleware.wrap_tool_call(unrelated, handler).content == "Results"


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def search_call(call_id, query="docs"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "langchainDocs_search_docs_by_lang_chain",
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_agent_deduplicates_across_summary_and_resets_on_new_turn(asynchronous):
    executions = []

    @tool
    def langchainDocs_search_docs_by_lang_chain(query: str) -> str:
        """Search the documentation."""
        executions.append(query)
        return "Documentation result " * 100

    model = ToolCallingModel(
        responses=[
            search_call("first"),
            search_call("repeat"),
            AIMessage(content="Done"),
            search_call("new-turn"),
            AIMessage(content="Done again"),
        ]
    )
    summary_model = FakeMessagesListChatModel(responses=[AIMessage(content="Summary")])
    agent = create_agent(
        model,
        tools=[langchainDocs_search_docs_by_lang_chain],
        middleware=[
            SummarizationMiddleware(
                model=summary_model, trigger=("messages", 3), keep=("messages", 1)
            ),
            DuplicateSearchMiddleware(),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "dedup"}}

    def invoke(text):
        inputs = {"messages": [HumanMessage(content=text)]}
        return (
            asyncio.run(agent.ainvoke(inputs, config))
            if asynchronous
            else agent.invoke(inputs, config)
        )

    result = invoke("Search docs")
    assert executions == ["docs"]
    assert result["messages"][-1].content == "Done"
    assert any("Summary" in str(msg.content) for msg in result["messages"])
    invoke("Search again")
    assert executions == ["docs", "docs"]


@pytest.mark.parametrize("failed_attempts", [1, 3])
def test_async_agent_preserves_retries_and_isolates_concurrent_invocations(
    failed_attempts,
):
    executions = []

    @tool
    async def langchainDocs_search_docs_by_lang_chain(query: str) -> str:
        """Search the documentation."""
        executions.append(query)
        await asyncio.sleep(0)
        if len(executions) <= failed_attempts:
            raise TimeoutError("timed out")
        return "Documentation result"

    middleware = DuplicateSearchMiddleware()

    def build_agent():
        return create_agent(
            ToolCallingModel(
                responses=[
                    search_call("first"),
                    search_call("repeat"),
                    search_call("repeat-after-recovery"),
                    AIMessage(content="Done"),
                ]
            ),
            tools=[langchainDocs_search_docs_by_lang_chain],
            middleware=[middleware, ToolRetryMiddleware(initial_delay=0)],
        )

    async def run():
        first = await build_agent().ainvoke(
            {"messages": [HumanMessage(content="Search")]}
        )
        assert first["messages"][-1].content == "Done"
        assert len(executions) == failed_attempts + 1
        await asyncio.gather(
            *[
                build_agent().ainvoke({"messages": [HumanMessage(content="Search")]})
                for _ in range(2)
            ]
        )

    asyncio.run(run())
    assert len(executions) == failed_attempts + 3


def test_guard_is_registered_before_retry(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from agent import docs_agent_middleware

    types = [type(middleware) for middleware in docs_agent_middleware]
    assert types.index(DuplicateSearchMiddleware) < types.index(ToolRetryMiddleware)
