"""Tests for request-only handling of cancelled user turns."""

import asyncio
from copy import deepcopy

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.latest_user_message_middleware import LatestUserMessageMiddleware


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [AIMessage(content="Hello")],
        [HumanMessage(content="One question")],
        [
            HumanMessage(content="First question"),
            AIMessage(content="First answer"),
            HumanMessage(content="Second question"),
        ],
    ],
)
def test_unambiguous_history_is_unchanged(messages, asynchronous):
    snapshot = deepcopy(messages)
    request = ModelRequest(
        model=object(), messages=messages, state={"messages": messages}
    )
    response = ModelResponse(result=[AIMessage(content="Answer")])
    calls = []

    def handler(marked):
        calls.append(marked)
        return response

    async def async_handler(marked):
        return handler(marked)

    middleware = LatestUserMessageMiddleware()
    if asynchronous:
        result = asyncio.run(middleware.awrap_model_call(request, async_handler))
    else:
        result = middleware.wrap_model_call(request, handler)

    assert result is response
    assert calls == [request]
    assert calls[0] is request
    assert request.messages is messages
    assert request.state["messages"] is messages
    assert messages == snapshot


@pytest.mark.parametrize("human_count", [2, 3])
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("structured_content", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_latest_human_is_current_without_mutating_state(
    human_count, with_tools, structured_content, asynchronous
):
    questions = ["Is A/B testing only possible for offline evals?"]
    if human_count == 3:
        questions.append("How do I create a dataset?")
    questions.append("What is the difference between score and value?")
    humans = [
        HumanMessage(
            content=[{"type": "text", "text": question}]
            if structured_content
            else question,
            id=f"human-{index}",
        )
        for index, question in enumerate(questions)
    ]
    suffix = (
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
            ),
            ToolMessage(content="Retrieved docs", tool_call_id="call-1"),
        ]
        if with_tools
        else []
    )
    messages = [AIMessage(content="Previous answer"), *humans, *suffix]
    snapshot = deepcopy(messages)
    state = {"messages": messages}
    request = ModelRequest(model=object(), messages=messages, state=state)
    calls = []
    response = ModelResponse(result=[AIMessage(content="Answer")])

    def handler(marked):
        calls.append(marked)
        return response

    async def async_handler(marked):
        return handler(marked)

    middleware = LatestUserMessageMiddleware()
    for _ in range(2):
        result = (
            asyncio.run(middleware.awrap_model_call(request, async_handler))
            if asynchronous
            else middleware.wrap_model_call(request, handler)
        )
        assert result is response

    for marked in calls:
        assert marked is not request
        assert marked.messages is not messages
        assert marked.state is state
        assert marked.messages[0] is messages[0]
        for index, human in enumerate(humans, start=1):
            copied = marked.messages[index]
            assert copied is not human
            assert copied.id == human.id
            marker = copied.content[0]["text"] if structured_content else copied.content
            if index == human_count:
                assert "Current user request: answer only this message" in marker
                assert questions[-1] in str(copied.content)
            else:
                assert "Cancelled or superseded user request: context only" in marker
            if structured_content:
                assert copied.content[1:] == human.content
        assert all(
            copied is original
            for copied, original in zip(marked.messages[human_count + 1 :], suffix)
        )

    assert request.messages is messages
    assert state["messages"] is messages
    assert messages == snapshot


def test_guard_is_reapplied_after_tool_execution():
    messages = [
        HumanMessage(content="Dropped question"),
        HumanMessage(content="Latest question"),
    ]
    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="Answer")])

    middleware = LatestUserMessageMiddleware()
    asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=messages), handler
        )
    )
    tool_call = AIMessage(
        content="", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}]
    )
    tool_result = ToolMessage(content="Docs", tool_call_id="call-1")
    after_tools = [*messages, tool_call, tool_result]
    asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=after_tools), handler
        )
    )

    assert calls[0].messages[:2] == calls[1].messages[:2]
    assert "Current user request" in calls[1].messages[1].content
    assert calls[1].messages[2] is tool_call
    assert calls[1].messages[3] is tool_result
    assert messages[0].content == "Dropped question"
    assert messages[1].content == "Latest question"
