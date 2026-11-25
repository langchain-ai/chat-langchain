"""Tests for tool-call history repair in the model fallback middleware."""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.middleware.model_fallback_middleware import (
    MISSING_TOOL_RESULT_PLACEHOLDER,
    RepairingModelFallbackMiddleware,
    repair_tool_call_ids,
)
from src.middleware.retry_middleware import ModelRetryMiddleware


def _ai_with_tool_calls(*ids):
    return AIMessage(
        content="",
        tool_calls=[{"id": tc_id, "name": "search", "args": {}} for tc_id in ids],
    )


def test_repair_inserts_missing_tool_message_when_partial():
    """An AIMessage with two tool_calls but only one ToolMessage gets one synthetic message."""
    messages = [
        HumanMessage(content="hi"),
        _ai_with_tool_calls("call_a", "call_b"),
        ToolMessage(tool_call_id="call_a", content="result a"),
        HumanMessage(content="next turn"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert len(repaired) == len(messages) + 1
    inserted = repaired[3]
    assert isinstance(inserted, ToolMessage)
    assert inserted.tool_call_id == "call_b"
    assert inserted.content == MISSING_TOOL_RESULT_PLACEHOLDER
    assert isinstance(repaired[4], HumanMessage)


def test_repair_inserts_all_when_block_is_fully_dangling():
    """An AIMessage with tool_calls followed by no ToolMessages gets all synthetics."""
    messages = [
        _ai_with_tool_calls("call_x", "call_y"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert len(repaired) == 3
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_msgs} == {"call_x", "call_y"}
    for tm in tool_msgs:
        assert tm.content == MISSING_TOOL_RESULT_PLACEHOLDER


def test_repair_is_noop_when_all_tool_calls_responded():
    """A clean history with every tool_call_id answered must be unchanged."""
    messages = [
        HumanMessage(content="hi"),
        _ai_with_tool_calls("call_a", "call_b"),
        ToolMessage(tool_call_id="call_a", content="result a"),
        ToolMessage(tool_call_id="call_b", content="result b"),
        AIMessage(content="done"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert len(repaired) == len(messages)
    for original, result in zip(messages, repaired):
        assert original is result


def test_repair_handles_multiple_tool_call_blocks():
    """Each AIMessage tool_calls block is repaired independently."""
    messages = [
        _ai_with_tool_calls("call_1"),
        # missing ToolMessage for call_1
        HumanMessage(content="continue"),
        _ai_with_tool_calls("call_2"),
        ToolMessage(tool_call_id="call_2", content="ok"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert isinstance(repaired[1], ToolMessage)
    assert repaired[1].tool_call_id == "call_1"
    assert repaired[1].content == MISSING_TOOL_RESULT_PLACEHOLDER
    # second block should be untouched
    assert isinstance(repaired[3], AIMessage)
    assert isinstance(repaired[4], ToolMessage)
    assert repaired[4].tool_call_id == "call_2"


def test_repair_ignores_ai_message_without_tool_calls():
    """An AIMessage without tool_calls is left alone."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="just a reply"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert repaired == messages


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "ending", ["answer", "human", "tool_call", "tool_result", "empty"]
)
def test_completed_turn_reuses_answer_without_invoking_models(asynchronous, ending):
    answer = AIMessage(content="Previous answer", id="answer-1")
    messages = [HumanMessage(content="Question", id="question-1"), answer]
    if ending == "human":
        messages.append(HumanMessage(content="Question", id="question-2"))
    elif ending == "tool_call":
        messages.append(_ai_with_tool_calls("call-1"))
    elif ending == "tool_result":
        messages.extend(
            [
                _ai_with_tool_calls("call-1"),
                ToolMessage(content="Result", tool_call_id="call-1"),
            ]
        )
    elif ending == "empty":
        messages = []
    model = FakeMessagesListChatModel(responses=[AIMessage(content="Fresh answer")])
    middleware = RepairingModelFallbackMiddleware(model)
    request = ModelRequest(model=model, messages=messages)
    calls = []
    fresh = ModelResponse(result=[AIMessage(content="Fresh answer")])

    def handler(request):
        calls.append(request)
        return fresh

    async def async_handler(request):
        return handler(request)

    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    if ending == "answer":
        assert result.result == [answer]
        assert result.result[0].id == "answer-1"
        assert calls == []
    else:
        assert result is fresh
        assert len(calls) == 1


def test_checkpointed_resend_reuses_answer_and_fresh_turn_calls_primary():
    primary = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="First answer", id="answer-1"),
            AIMessage(content="Fresh answer", id="answer-2"),
        ]
    )
    fallback = FakeMessagesListChatModel(
        responses=[AIMessage(content="Fallback answer")]
    )
    graph = create_agent(
        model=primary,
        middleware=[
            RepairingModelFallbackMiddleware(fallback),
            ModelRetryMiddleware(initial_delay=0),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "resend"}}
    question = HumanMessage(content="Question", id="question-1")

    async def run():
        async def invoke(inputs):
            return await graph.ainvoke(inputs, config)

        first = await invoke({"messages": [question]})
        replay = await invoke({"messages": first["messages"]})
        duplicate = await invoke({"messages": [question]})
        fresh = await invoke(
            {"messages": [HumanMessage(content="Question", id="question-2")]}
        )
        return first, replay, duplicate, fresh

    first, replay, duplicate, fresh = asyncio.run(run())
    assert [message.id for message in first["messages"]] == ["question-1", "answer-1"]
    assert replay["messages"] == first["messages"]
    assert duplicate["messages"] == first["messages"]
    assert [message.id for message in fresh["messages"]] == [
        "question-1",
        "answer-1",
        "question-2",
        "answer-2",
    ]
    assert fresh["messages"][-1].content == "Fresh answer"


def _gemini_history():
    message = _ai_with_tool_calls("call_a")
    message.additional_kwargs = {
        "function_call": {"name": "search", "arguments": "{}"},
        "__gemini_function_call_thought_signatures__": {"call_a": "signature"},
        "other": "preserved",
    }
    return [
        HumanMessage(content="hi"),
        message,
        ToolMessage(tool_call_id="call_a", content="result"),
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("experiment", [False, True])
@pytest.mark.parametrize("fireworks_fallback", [False, True])
def test_fireworks_history_payload(asynchronous, experiment, fireworks_fallback):
    from langchain_fireworks import ChatFireworks
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.middleware.model_experiment_middleware import (
        ARM_BASE,
        ModelChain,
        ModelExperiment,
        ModelExperimentFallbackMiddleware,
    )

    fireworks = ChatFireworks(model="accounts/fireworks/models/test", api_key="test")
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="test")
    primary, fallback = (
        (gemini, fireworks) if fireworks_fallback else (fireworks, gemini)
    )
    if experiment:
        chain = ModelChain(primary, (fallback,))
        middleware = ModelExperimentFallbackMiddleware(
            ModelExperiment(
                name="test",
                control=ModelChain(gemini, (gemini,)),
                treatment=chain,
                base=chain,
                treatment_percent=0,
                base_percent=100,
            )
        )
    else:
        middleware = RepairingModelFallbackMiddleware(fallback, primary_model=primary)
    messages = _gemini_history()
    request = ModelRequest(
        model=gemini, messages=messages, state={"model_arm": ARM_BASE}
    )
    targets = []
    response = ModelResponse(result=[AIMessage(content="answer")])

    requests = []

    def handler(current):
        targets.append(current.model)
        requests.append(current)
        if len(targets) == 1:
            raise RuntimeError("primary unavailable")
        return response

    async def async_handler(current):
        return handler(current)

    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert result is response
    assert targets == [primary, fallback]
    for current in requests:
        if current.model is gemini:
            assert current.messages[1] is messages[1]
        else:
            payload, _ = fireworks._create_message_dicts(current.messages, stop=None)
            assistant = payload[1]
            assert "function_call" not in assistant
            assert "__gemini_function_call_thought_signatures__" not in assistant
            assert assistant["tool_calls"] == [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ]
            assert current.messages[1].additional_kwargs == {"other": "preserved"}
            assert current.messages[2] is messages[2]

    assert "function_call" in messages[1].additional_kwargs
    assert messages[1].tool_calls == _ai_with_tool_calls("call_a").tool_calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_clean_primary_history_preserves_identity(asynchronous):
    model = FakeMessagesListChatModel(responses=[AIMessage(content="answer")])
    middleware = RepairingModelFallbackMiddleware(model)
    messages = [
        HumanMessage(content="hi"),
        _ai_with_tool_calls("call_a"),
        ToolMessage(tool_call_id="call_a", content="result"),
    ]
    request = ModelRequest(model=model, messages=messages)
    response = ModelResponse(result=[AIMessage(content="answer")])

    def handler(current):
        assert current.messages is request.messages
        assert all(a is b for a, b in zip(current.messages, messages))
        return response

    async def async_handler(current):
        return handler(current)

    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert result is response


@pytest.mark.parametrize("asynchronous", [False, True])
def test_fireworks_succeeds_after_gemini_tool_call_fallback(monkeypatch, asynchronous):
    from langchain_fireworks import ChatFireworks
    from langchain_google_genai import ChatGoogleGenerativeAI

    fireworks = ChatFireworks(model="accounts/fireworks/models/test", api_key="test")
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="test")
    middleware = RepairingModelFallbackMiddleware(gemini, primary_model=fireworks)
    payloads = []

    def create(**payload):
        payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }

    async def acreate(**payload):
        return create(**payload)

    monkeypatch.setattr(fireworks.client, "create", create)
    monkeypatch.setattr(fireworks.async_client, "create", acreate)
    history = _gemini_history()
    calls = []

    def handler(request):
        calls.append(request.model)
        if len(calls) == 1:
            raise RuntimeError("initial Fireworks failure")
        if request.model is gemini:
            return ModelResponse(result=[history[1]])
        return ModelResponse(result=[request.model.invoke(request.messages)])

    async def async_handler(request):
        if len(calls) < 2:
            return handler(request)
        calls.append(request.model)
        return ModelResponse(result=[await request.model.ainvoke(request.messages)])

    async def run():
        first = await middleware.awrap_model_call(
            ModelRequest(model=fireworks, messages=history[:1]), async_handler
        )
        return await middleware.awrap_model_call(
            ModelRequest(
                model=fireworks, messages=[history[0], first.result[0], history[2]]
            ),
            async_handler,
        )

    if asynchronous:
        result = asyncio.run(run())
    else:
        first = middleware.wrap_model_call(
            ModelRequest(model=fireworks, messages=history[:1]), handler
        )
        result = middleware.wrap_model_call(
            ModelRequest(
                model=fireworks, messages=[history[0], first.result[0], history[2]]
            ),
            handler,
        )
    assert result.result[0].content == "answer"
    assert calls == [fireworks, gemini, fireworks]
    assert len(payloads) == 1
    assert "function_call" not in payloads[0]["messages"][1]
    assert payloads[0]["messages"][1]["tool_calls"][0]["id"] == "call_a"
