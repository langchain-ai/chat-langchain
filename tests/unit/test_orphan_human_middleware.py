"""Tests for request-only handling of superseded user messages."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.orphan_human_middleware import OrphanHumanMiddleware


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def prepare_request(request):
    middleware = OrphanHumanMiddleware()

    def invoke(model_request):
        captured = []
        response = ModelResponse(result=[AIMessage(content="Latest answer")])

        def handler(next_request):
            captured.append(next_request)
            return response

        async def async_handler(next_request):
            return handler(next_request)

        if request.param:
            result = asyncio.run(
                middleware.awrap_model_call(model_request, async_handler)
            )
        else:
            result = middleware.wrap_model_call(model_request, handler)
        assert result is response
        return captured[0]

    return invoke


def test_orphan_is_marked_without_mutating_state(prepare_request):
    messages = [
        HumanMessage(content="A", id="a"),
        AIMessage(content="Answer A"),
        HumanMessage(content="B", id="b"),
        HumanMessage(content="C", id="c"),
    ]
    state = {"messages": messages}
    request = ModelRequest(model=object(), messages=messages, state=state)

    prepared = prepare_request(request)

    assert prepared.messages[:2] == messages[:2]
    assert prepared.messages[2].content == (
        "[cancelled by user before a reply - superseded]\nB"
    )
    assert prepared.messages[2].id == "b"
    assert prepared.messages[-1] is messages[-1]
    assert request.messages is messages
    assert state["messages"][2].content == "B"
    assert prepared.state is state


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [HumanMessage(content="Current")],
        [
            HumanMessage(content="A"),
            AIMessage(content="Answer A"),
            HumanMessage(content="C"),
        ],
    ],
)
def test_no_orphan_passes_through_unchanged(prepare_request, messages):
    request = ModelRequest(model=object(), messages=messages)

    assert prepare_request(request) is request


def test_tool_call_pairs_remain_unchanged(prepare_request):
    messages = [
        HumanMessage(content="A"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
        ),
        ToolMessage(content="Docs", tool_call_id="call-1"),
        HumanMessage(content="B"),
        HumanMessage(content="C"),
    ]

    prepared = prepare_request(ModelRequest(model=object(), messages=messages))

    assert all(prepared.messages[index] is messages[index] for index in range(3))
    assert prepared.messages[-1] is messages[-1]
    assert prepared.messages[3].content.startswith("[cancelled by user")


def test_all_consecutive_orphans_are_marked_with_multimodal_content(prepare_request):
    content = [
        {"type": "text", "text": "Earlier question"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
    messages = [
        HumanMessage(content=content),
        HumanMessage(content="Also abandoned"),
        HumanMessage(content="Current question"),
    ]

    prepared = prepare_request(ModelRequest(model=object(), messages=messages))

    assert prepared.messages[0].content[0]["text"].startswith("[cancelled by user")
    assert prepared.messages[0].content[1:] == content
    assert prepared.messages[1].content.startswith("[cancelled by user")
    assert prepared.messages[-1] is messages[-1]
    assert messages[0].content == content
