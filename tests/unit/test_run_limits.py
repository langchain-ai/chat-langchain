"""Tests for server-side model-call limits independent of client config."""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent import docs_agent_middleware


class _LoopingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("config", [None, {"recursion_limit": 1000}])
@pytest.mark.parametrize("async_run", [False, True])
def test_server_limit_applies_without_client_limit(config, async_run):
    limit = next(
        middleware
        for middleware in docs_agent_middleware
        if isinstance(middleware, ModelCallLimitMiddleware)
    )
    assert limit.run_limit == 100
    assert limit.thread_limit is None
    assert limit.exit_behavior == "error"
    calls = []

    @tool
    def repeat() -> str:
        """Continue the test loop."""
        calls.append("tool")
        return "continue"

    model = _LoopingModel(
        responses=[
            AIMessage(
                id=f"model-{index}",
                content="",
                tool_calls=[{"name": "repeat", "args": {}, "id": f"repeat-{index}"}],
            )
            for index in range(101)
        ]
    )
    graph = create_agent(model, tools=[repeat], middleware=[limit])
    inputs = {"messages": [HumanMessage(content="Loop")]}

    with pytest.raises(ModelCallLimitExceededError) as error:
        if async_run:
            asyncio.run(graph.ainvoke(inputs, config))
        else:
            graph.invoke(inputs, config)

    assert len(calls) == 100
    assert error.value.run_count == 100


def test_completed_guest_runs_do_not_accumulate_toward_run_limit():
    limit = next(
        middleware
        for middleware in docs_agent_middleware
        if isinstance(middleware, ModelCallLimitMiddleware)
    )
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="Hello", id=f"response-{index}") for index in range(101)
        ]
    )
    graph = create_agent(model, middleware=[limit], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "guest-thread"}}

    for index in range(101):
        result = graph.invoke(
            {"messages": [HumanMessage(content=f"Hello {index}")]}, config
        )
        assert result["messages"][-1].content == "Hello"
