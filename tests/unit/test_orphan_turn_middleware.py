"""Tests for model-only disambiguation of unanswered user turns."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.middleware.orphan_turn_middleware import OrphanHumanMessageMiddleware


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def model_call(request):
    def invoke(messages):
        original = ModelRequest(
            model=object(), messages=messages, state={"messages": messages}
        )
        snapshot = [message.model_copy(deep=True) for message in messages]
        calls = []
        response = ModelResponse(result=[AIMessage(content="Answer")])

        def handler(model_request):
            calls.append(model_request)
            return response

        async def async_handler(model_request):
            return handler(model_request)

        middleware = OrphanHumanMessageMiddleware()
        if request.param:
            result = asyncio.run(middleware.awrap_model_call(original, async_handler))
        else:
            result = middleware.wrap_model_call(original, handler)

        assert result is response
        assert len(calls) == 1
        assert original.messages is messages
        assert original.state["messages"] is messages
        assert messages == snapshot
        assert calls[0].state is original.state
        return original, calls[0]

    return invoke


@pytest.mark.parametrize("orphan_count", [1, 2])
def test_unanswered_questions_are_folded_into_context(model_call, orphan_count):
    answered = [HumanMessage(content="A"), AIMessage(content="Answer to A")]
    orphans = [HumanMessage(content=f"Orphan {index}") for index in range(orphan_count)]
    latest = HumanMessage(content="Latest question", id="latest", name="user")
    original, prepared = model_call([*answered, *orphans, latest])

    assert prepared is not original
    assert prepared.messages[:2] == answered
    assert len(prepared.messages) == 4
    assert prepared.messages[2].content == (
        "Earlier messages sent without an answer "
        "(context only; superseded unless the current question refers to them):\n\n"
        + "\n\n".join(
            f"{index + 1}. {orphan.content}" for index, orphan in enumerate(orphans)
        )
    )
    assert (
        prepared.messages[3].content == "Current question to answer:\n\nLatest question"
    )
    assert prepared.messages[3].id == latest.id
    assert prepared.messages[3].name == latest.name
    assert prepared.messages[3] is not latest


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [SystemMessage(content="Instructions")],
        [HumanMessage(content="Latest")],
        [
            HumanMessage(content="A"),
            AIMessage(content="Answer"),
            HumanMessage(content="C"),
        ],
        [
            HumanMessage(content="A"),
            AIMessage(content=[{"type": "text", "text": "Answer"}]),
            HumanMessage(content="C"),
        ],
    ],
)
def test_requests_without_orphans_pass_through_unchanged(model_call, messages):
    original, prepared = model_call(messages)

    assert prepared is original


@pytest.mark.parametrize(
    "content", ["", "  ", [{"type": "thinking", "thinking": "Researching"}]]
)
def test_tool_calls_do_not_answer_earlier_questions(model_call, content):
    tool_call = AIMessage(
        content=content,
        tool_calls=[{"name": "search", "args": {}, "id": "search-1"}],
    )
    tool_result = ToolMessage(content="Docs", tool_call_id="search-1")
    current_call = AIMessage(
        content="", tool_calls=[{"name": "search", "args": {}, "id": "search-2"}]
    )
    current_result = ToolMessage(content="New docs", tool_call_id="search-2")
    _, prepared = model_call(
        [
            HumanMessage(content="Orphan"),
            tool_call,
            tool_result,
            HumanMessage(content="Latest"),
            current_call,
            current_result,
        ]
    )

    assert "1. Orphan" in prepared.messages[0].content
    assert prepared.messages[1:3] == [tool_call, tool_result]
    assert prepared.messages[3].content == "Current question to answer:\n\nLatest"
    assert prepared.messages[4:] == [current_call, current_result]


def test_multimodal_current_question_keeps_attachments(model_call):
    content = [
        {"type": "text", "text": "Explain this diagram"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
    _, prepared = model_call(
        [
            HumanMessage(content=[{"type": "text", "text": "Earlier question"}]),
            HumanMessage(content=content, id="latest"),
        ]
    )

    assert "1. Earlier question" in prepared.messages[0].content
    assert prepared.messages[1].content == [
        {"type": "text", "text": "Current question to answer:"},
        *content,
    ]
