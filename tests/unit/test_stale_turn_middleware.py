"""Tests for request-only annotations of superseded user messages."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.stale_turn_middleware import StaleTurnMiddleware

PREFIX = (
    "[Earlier message from a stopped turn—the user moved on; "
    "use only as context if the latest message refers to it]"
)


def _invoke(middleware, request, asynchronous):
    calls = []
    response = ModelResponse(result=[AIMessage(content="Latest answer")])

    def handler(model_request):
        calls.append(model_request)
        return response

    async def async_handler(model_request):
        return handler(model_request)

    if asynchronous:
        result = asyncio.run(middleware.awrap_model_call(request, async_handler))
    else:
        result = middleware.wrap_model_call(request, handler)

    assert result is response
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [AIMessage(content="Prior answer")],
        [HumanMessage(content="Latest question")],
        [
            HumanMessage(content="Earlier question"),
            AIMessage(content="Prior answer"),
            HumanMessage(content="Latest question"),
        ],
        [
            HumanMessage(content="Earlier stopped question"),
            HumanMessage(content="Answered question"),
            AIMessage(content="Prior answer"),
            HumanMessage(content="Latest question"),
        ],
    ],
)
def test_no_stale_messages_leave_request_unchanged(messages, asynchronous):
    request = ModelRequest(model=object(), messages=messages)

    assert _invoke(StaleTurnMiddleware(), request, asynchronous) is request


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stale_count", [1, 2, 3])
@pytest.mark.parametrize("prior_answer", [False, True])
@pytest.mark.parametrize("tool_results", [False, True])
def test_all_stale_messages_are_marked_without_changing_state(
    asynchronous, stale_count, prior_answer, tool_results
):
    earlier = (
        [HumanMessage(content="Answered question"), AIMessage(content="Prior answer")]
        if prior_answer
        else []
    )
    stale = [
        HumanMessage(
            content=f"Stopped question {index}",
            id=f"stopped-{index}",
            additional_kwargs={"source": "chat"},
        )
        for index in range(stale_count)
    ]
    latest = HumanMessage(content="Latest question", id="latest")
    subsequent = (
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_docs", "args": {}, "id": "search-1"}],
            ),
            ToolMessage(content="Docs result", tool_call_id="search-1"),
        ]
        if tool_results
        else []
    )
    messages = [*earlier, *stale, latest, *subsequent]
    original_messages = [message.model_copy(deep=True) for message in messages]
    state = {"messages": messages}
    request = ModelRequest(model=object(), messages=messages, state=state)
    middleware = StaleTurnMiddleware()

    for _ in range(2):
        annotated = _invoke(middleware, request, asynchronous)

        assert annotated is not request
        assert annotated.state is state
        assert len(annotated.messages) == len(messages)
        for index, message in enumerate(messages):
            actual = annotated.messages[index]
            if len(earlier) <= index < len(earlier) + stale_count:
                assert actual is not message
                assert actual.content == f"{PREFIX}\n{message.content}"
                assert actual.id == message.id
                assert actual.additional_kwargs == message.additional_kwargs
            else:
                assert actual is message
        assert request.messages == original_messages
        assert state["messages"] == original_messages


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stale_content_blocks_are_preserved_without_mutation(asynchronous):
    stale = HumanMessage(
        content=[
            {"type": "text", "text": "Stopped question"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image"}},
        ]
    )
    latest = HumanMessage(content=[{"type": "text", "text": "Latest question"}])
    request = ModelRequest(model=object(), messages=[stale, latest])
    original = stale.model_copy(deep=True)

    annotated = _invoke(StaleTurnMiddleware(), request, asynchronous)

    assert annotated.messages[0].content == [
        {"type": "text", "text": PREFIX},
        *original.content,
    ]
    assert annotated.messages[1] is latest
    annotated.messages[0].content[1]["text"] = "Changed request copy"
    assert stale == original
