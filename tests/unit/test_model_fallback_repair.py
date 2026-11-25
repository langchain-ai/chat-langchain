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
    return [
        HumanMessage(content="Search docs"),
        AIMessage(
            content="Searching",
            id="gemini-tool-message",
            tool_calls=[{"id": "call-1", "name": "search", "args": {"query": "docs"}}],
            additional_kwargs={
                "function_call": {"name": "search", "arguments": '{"query":"docs"}'},
                "__gemini_function_call_thought_signatures__": {"call-1": "signature"},
                "other": "preserved",
            },
        ),
        ToolMessage(content="Docs result", tool_call_id="call-1"),
    ]


@pytest.mark.parametrize(
    "messages", [[], [HumanMessage(content="hi")], [_ai_with_tool_calls("a")]]
)
def test_sanitize_clean_history_retains_list(messages):
    from src.middleware.model_fallback_middleware import sanitize_gemini_metadata

    assert sanitize_gemini_metadata(messages) is messages


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("experiment", [False, True])
@pytest.mark.parametrize("gemini_first", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.parametrize("fail_primary", [False, True])
def test_each_target_gets_compatible_history(
    asynchronous, experiment, gemini_first, dangling, fail_primary
):
    from langchain_fireworks import ChatFireworks
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.middleware.model_experiment_middleware import (
        ARM_TREATMENT,
        ModelChain,
        ModelExperiment,
        ModelExperimentFallbackMiddleware,
    )

    fireworks = ChatFireworks(model="accounts/fireworks/models/test", api_key="test")
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="test")
    primary, fallback = (gemini, fireworks) if gemini_first else (fireworks, gemini)
    if experiment:
        chain = ModelChain(primary, (fallback,))
        middleware = ModelExperimentFallbackMiddleware(
            ModelExperiment(
                name="test",
                control=chain,
                treatment=chain,
                treatment_percent=100,
                treatment_instruction="Be concise.",
            )
        )
    else:
        middleware = RepairingModelFallbackMiddleware(fallback, primary_model=primary)
    messages = _gemini_history()
    if dangling:
        messages.pop()
    original = messages[1].model_copy(deep=True)
    request = ModelRequest(
        model=fallback, messages=messages, state={"model_arm": ARM_TREATMENT}
    )
    calls = []
    response = ModelResponse(result=[AIMessage(content="answer")])

    def handler(outgoing):
        calls.append(outgoing.model)
        ai = outgoing.messages[1]
        assert ai.content == original.content
        assert ai.id == original.id
        assert ai.tool_calls == original.tool_calls
        assert ai.additional_kwargs.get("other") == "preserved"
        assert outgoing.messages[2].tool_call_id == "call-1"
        if outgoing.model is gemini:
            assert ai.additional_kwargs == original.additional_kwargs
        else:
            assert ai.additional_kwargs == {"other": "preserved"}
            wire, _ = fireworks._create_message_dicts(outgoing.messages, None)
            assert "function_call" not in wire[1]
            assert "__gemini_function_call_thought_signatures__" not in wire[1]
            assert wire[1]["tool_calls"][0]["id"] == "call-1"
        if experiment:
            assert outgoing.system_message.content[-1]["text"] == "Be concise."
        if fail_primary and len(calls) == 1:
            raise TimeoutError("model timeout")
        return response

    async def async_handler(outgoing):
        return handler(outgoing)

    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert result is response
    assert calls == ([primary, fallback] if fail_primary else [primary])
    assert messages[1] == original
    assert len(messages) == (2 if dangling else 3)


def test_fireworks_arm_recovers_after_gemini_tool_fallback(monkeypatch):
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_fireworks import ChatFireworks
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.middleware.model_experiment_middleware import (
        ARM_TREATMENT,
        ModelChain,
        ModelExperiment,
        ModelExperimentFallbackMiddleware,
    )

    fireworks = ChatFireworks(model="accounts/fireworks/models/test", api_key="test")
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="test")
    payloads = []

    def complete(llm, **kwargs):
        payloads.append(kwargs["messages"])
        if len(payloads) == 1:
            raise TimeoutError("tool-use timeout")
        for message in kwargs["messages"]:
            assert "function_call" not in message
            assert "__gemini_function_call_thought_signatures__" not in message
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Fireworks answer"},
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(
        "langchain_fireworks.chat_models._completion_with_retry", complete
    )
    monkeypatch.setattr(
        ChatGoogleGenerativeAI,
        "_generate",
        lambda *a, **kw: ChatResult(
            generations=[ChatGeneration(message=_gemini_history()[1])]
        ),
    )

    def search(query: str) -> str:
        """Search documentation."""
        return "Docs result"

    chain = ModelChain(fireworks, (gemini,))
    graph = create_agent(
        model=fireworks,
        tools=[search],
        middleware=[
            ModelExperimentFallbackMiddleware(
                ModelExperiment(
                    name="test",
                    control=chain,
                    treatment=chain,
                    treatment_percent=100,
                    treatment_instruction="Be concise.",
                )
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "gemini-fallback"}, "recursion_limit": 10}
    graph.update_state(config, {"model_arm": ARM_TREATMENT})
    first = graph.invoke({"messages": [HumanMessage(content="Search docs")]}, config)
    second = graph.invoke({"messages": [HumanMessage(content="Follow up")]}, config)
    assert first["messages"][-1].content == "Fireworks answer"
    assert second["messages"][-1].content == "Fireworks answer"
    assert len(payloads) == 3
    tool_call = next(m for m in payloads[1] if m.get("tool_calls"))
    assert tool_call["tool_calls"][0]["id"] == "call-1"
    stored = next(m for m in second["messages"] if m.id == "gemini-tool-message")
    assert stored.additional_kwargs == _gemini_history()[1].additional_kwargs
