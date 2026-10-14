import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    RECOVERY_MESSAGE,
    MalformedResponseError,
    ModelRecoveryMiddleware,
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
    validate_model_response,
)


def test_provider_validation_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    with pytest.raises(ValueError, match="unsupported request shape"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "MALFORMED_FUNCTION_CALL"},
        {"finish_reason": "MAX_TOKENS"},
        {"finish_reason": "length"},
        {"stop_reason": "max_tokens"},
    ],
)
def test_rejects_provider_length_limits_on_message_metadata(metadata, wrapped):
    message = AIMessage(content="Truncated answer", response_metadata=metadata)
    response = ModelResponse(result=[message]) if wrapped else message
    with pytest.raises(MalformedResponseError):
        validate_model_response(response)


@pytest.mark.parametrize(
    "content",
    [
        "Explain the program\n" + "*   " * 5000,
        [{"type": "text", "text": "Title\n" + "*   " * 5000}],
        ["Title\n", {"type": "text", "text": "*   " * 5000}],
        [{"type": "text", "text": "*   " * 2500}] * 2,
        "Title\n" + "abcdefgh" * 1000,
    ],
)
def test_rejects_long_repetitive_stop_responses(content):
    message = AIMessage(content=content, response_metadata={"finish_reason": "STOP"})
    with pytest.raises(MalformedResponseError, match="repetitive"):
        validate_model_response(ModelResponse(result=[message]))


@pytest.mark.parametrize(
    "content",
    [
        "Use VectorStore.add_texts to add documents to the vector store.",
        [{"type": "text", "text": "This line adds texts."}],
        "Markdown separator: " + "-" * 60,
        "*   " * 100,
        "A long explanation of the API. " * 100 + "-" * 200,
        "The alphabet is repeated here: " + "abcdefgh" * 50,
    ],
)
def test_accepts_normal_answers_and_short_legitimate_repetition(content):
    message = AIMessage(content=content)
    assert validate_model_response(message) is message


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_retry_validates_every_attempt(asynchronous, exhausted):
    middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    bad = ModelResponse(result=[AIMessage(content="Title\n" + "*   " * 5000)])
    good = ModelResponse(result=[AIMessage(content="This line adds documents.")])
    responses = [bad, bad if exhausted else good]
    calls = []

    def handler(request):
        calls.append(request)
        return responses.pop(0)

    async def async_handler(request):
        return handler(request)

    request = ModelRequest(model=object(), messages=[HumanMessage(content="Explain")])

    def invoke():
        if asynchronous:
            return asyncio.run(middleware.awrap_model_call(request, async_handler))
        return middleware.wrap_model_call(request, handler)

    if exhausted:
        with pytest.raises(MalformedResponseError):
            invoke()
    else:
        assert invoke() is good
    assert len(calls) == 2


class CountingModel(FakeMessagesListChatModel):
    calls: int = 0
    validation_error: bool = False

    def _generate(self, *args, **kwargs):
        self.calls += 1
        if self.validation_error:
            raise ValueError("unsupported request shape")
        return super()._generate(*args, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("accepted_model", [0, 1, 2, None])
def test_graph_retries_each_model_then_falls_back_or_recovers(
    asynchronous, accepted_model
):
    bad = AIMessage(content="Title\n" + "*   " * 5000)
    good = AIMessage(
        content="VectorStore.add_texts embeds and stores each supplied text."
    )
    models = [
        CountingModel(
            responses=[good if index == accepted_model else bad], disable_streaming=True
        )
        for index in range(3)
    ]
    graph = create_agent(
        model=models[0],
        middleware=[
            ModelRecoveryMiddleware(models[0]),
            ModelFallbackMiddleware(*models[1:]),
            ModelRetryMiddleware(max_retries=1, initial_delay=0),
        ],
    )
    inputs = {"messages": [HumanMessage(content="Explain VectorStore.add_texts")]}
    result = (
        asyncio.run(graph.ainvoke(inputs)) if asynchronous else graph.invoke(inputs)
    )
    assert result["messages"][-1].content == (
        RECOVERY_MESSAGE if accepted_model is None else good.content
    )
    assert all(message.content != bad.content for message in result["messages"])
    expected_calls = [
        2
        if accepted_model is None or index < accepted_model
        else 1
        if index == accepted_model
        else 0
        for index in range(3)
    ]
    assert [model.calls for model in models] == expected_calls


def test_provider_validation_error_advances_to_fallback_without_retry():
    primary = CountingModel(
        responses=[AIMessage(content="unused")], validation_error=True
    )
    fallback = CountingModel(responses=[AIMessage(content="Accepted answer")])
    graph = create_agent(
        model=primary,
        middleware=[
            ModelRecoveryMiddleware(primary),
            ModelFallbackMiddleware(fallback),
            ModelRetryMiddleware(max_retries=1, initial_delay=0),
        ],
    )
    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="Explain")]}))
    assert result["messages"][-1].content == "Accepted answer"
    assert primary.calls == fallback.calls == 1


def test_recovery_replaces_the_unbounded_runtime_primary():
    runtime_model = CountingModel(responses=[AIMessage(content="Unbounded")])
    bounded_model = CountingModel(responses=[AIMessage(content="Bounded answer")])
    graph = create_agent(
        model=runtime_model,
        middleware=[ModelRecoveryMiddleware(bounded_model), ModelRetryMiddleware()],
    )
    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="Explain")]}))
    assert result["messages"][-1].content == "Bounded answer"
    assert runtime_model.calls == 0
    assert bounded_model.calls == 1


def test_streamed_graph_state_contains_only_accepted_answer():
    primary = CountingModel(responses=[AIMessage(content="Title\n" + "*   " * 5000)])
    fallback = CountingModel(responses=[AIMessage(content="Accepted answer")])
    graph = create_agent(
        model=primary,
        middleware=[
            ModelRecoveryMiddleware(primary),
            ModelFallbackMiddleware(fallback),
            ModelRetryMiddleware(max_retries=1, initial_delay=0),
        ],
    )

    async def collect():
        return [
            data
            async for mode, data in graph.astream(
                {"messages": [HumanMessage(content="Explain")]},
                stream_mode=["values", "updates"],
            )
            if mode == "values"
        ]

    states = asyncio.run(collect())
    answers = [
        message.content
        for state in states
        for message in state["messages"]
        if isinstance(message, AIMessage)
    ]
    assert answers == ["Accepted answer"]


def test_model_retry_wrapper_does_not_retry_provider_validation_error():
    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(ValueError, match="unsupported request shape"):
        runnable.invoke("request")

    assert calls == 1
