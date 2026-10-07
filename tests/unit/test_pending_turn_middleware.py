"""Tests for request-only guidance on unanswered human turns."""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.middleware.pending_turn_middleware import PendingTurnMiddleware

_NOTE = (
    "The user sent earlier messages that were cancelled before an answer. "
    "Answer ONLY the final user message; use earlier ones as context only."
)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("pending_count", [2, 3])
def test_pending_turns_add_transient_guidance(asynchronous, pending_count):
    messages = [AIMessage(content="Previous answer")]
    messages.extend(
        HumanMessage(content=question) for question in ["A", "B", "C"][:pending_count]
    )
    state = {"messages": messages}
    request = ModelRequest(
        model=object(),
        messages=messages,
        state=state,
        system_message=SystemMessage(content="Answer using documentation."),
    )
    response = ModelResponse(result=[AIMessage(content="Latest answer")])
    calls = []
    middleware = PendingTurnMiddleware()

    def handler(model_request):
        calls.append(model_request)
        return response

    async def async_handler(model_request):
        return handler(model_request)

    for _ in range(2):
        if asynchronous:
            result = asyncio.run(middleware.awrap_model_call(request, async_handler))
        else:
            result = middleware.wrap_model_call(request, handler)
        assert result is response

    for model_request in calls:
        assert model_request.system_message.content == (
            f"Answer using documentation.\n\n{_NOTE}"
        )
        assert model_request.messages == messages
        assert model_request.messages[-1].content == ["B", "C"][pending_count - 2]
        assert model_request.messages[1].content == "A"
        assert model_request.state == state
    assert request.system_message.content == "Answer using documentation."
    assert state["messages"] == messages
    assert len(messages) == pending_count + 1


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [HumanMessage(content="B")],
        [AIMessage(content="Previous answer"), HumanMessage(content="B")],
        [
            HumanMessage(content="A"),
            AIMessage(content="Answer"),
            HumanMessage(content="B"),
        ],
        [
            HumanMessage(content="A"),
            ToolMessage(content="Result", tool_call_id="tool"),
            HumanMessage(content="B"),
        ],
        [
            HumanMessage(content="A"),
            HumanMessage(content="B"),
            AIMessage(content="Answer"),
        ],
        [
            HumanMessage(content="A"),
            HumanMessage(content="B"),
            ToolMessage(content="Result", tool_call_id="tool"),
        ],
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_other_histories_are_unchanged(messages, asynchronous):
    request = ModelRequest(model=object(), messages=messages)
    response = ModelResponse(result=[AIMessage(content="Answer")])

    def handler(model_request):
        assert model_request is request
        return response

    async def async_handler(model_request):
        return handler(model_request)

    middleware = PendingTurnMiddleware()
    if asynchronous:
        result = asyncio.run(middleware.awrap_model_call(request, async_handler))
    else:
        result = middleware.wrap_model_call(request, handler)
    assert result is response


@pytest.mark.parametrize(
    "system_message",
    [None, SystemMessage(content=[{"type": "text", "text": "Existing instructions"}])],
)
def test_guidance_preserves_system_content_and_multimodal_user_message(system_message):
    messages = [
        HumanMessage(content="A"),
        HumanMessage(content=[{"type": "text", "text": "B"}]),
    ]
    request = ModelRequest(
        model=object(), messages=messages, system_message=system_message
    )

    def handler(model_request):
        assert model_request.messages == messages
        if system_message is None:
            assert model_request.system_message.content == _NOTE
        else:
            assert model_request.system_message.content == [
                *system_message.content,
                {"type": "text", "text": _NOTE},
            ]
        return ModelResponse(result=[AIMessage(content="Answer")])

    PendingTurnMiddleware().wrap_model_call(request, handler)
    assert request.system_message is system_message


def test_guidance_is_not_checkpointed_or_carried_into_next_turn():
    class CaptureModelInput(BaseCallbackHandler):
        def __init__(self):
            self.messages = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.messages.extend(messages)

    capture = CaptureModelInput()
    agent = create_agent(
        model=FakeMessagesListChatModel(
            responses=[AIMessage(content="Answer B"), AIMessage(content="Answer C")]
        ),
        middleware=[PendingTurnMiddleware()],
        system_prompt="Answer using documentation.",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "pending-turn"}, "callbacks": [capture]}
    agent.update_state(
        config,
        {"messages": [AIMessage(content="Previous answer"), HumanMessage(content="A")]},
    )

    agent.invoke({"messages": [HumanMessage(content="B")]}, config)

    assert capture.messages[0][0].content == f"Answer using documentation.\n\n{_NOTE}"
    assert [message.content for message in capture.messages[0][1:]] == [
        "Previous answer",
        "A",
        "B",
    ]
    assert [
        message.content for message in agent.get_state(config).values["messages"]
    ] == ["Previous answer", "A", "B", "Answer B"]

    agent.invoke({"messages": [HumanMessage(content="C")]}, config)

    assert capture.messages[1][0].content == "Answer using documentation."
    assert not any(
        isinstance(message, SystemMessage)
        for message in agent.get_state(config).values["messages"]
    )
