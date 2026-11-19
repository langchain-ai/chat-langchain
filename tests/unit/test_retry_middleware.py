"""Tests for model response retries and fallback exhaustion."""

import asyncio

import pytest
from langchain.agents.factory import _chain_async_model_call_handlers
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import MalformedResponseError, ModelRetryMiddleware


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        "",
        " \n\t",
        [],
        [{"type": "text", "text": ""}],
        [{"type": "text", "text": " \n"}],
    ],
)
def test_empty_stop_response_is_retried(content, wrapped):
    empty = AIMessage(content=content, response_metadata={"finish_reason": "STOP"})
    answer = AIMessage(content="Here is an answer")
    outcomes = [empty, answer]

    async def handler(request):
        message = outcomes.pop(0)
        return ModelResponse(result=[message]) if wrapped else message

    response = asyncio.run(
        ModelRetryMiddleware(initial_delay=0).awrap_model_call(None, handler)
    )

    assert (response.result[-1] if wrapped else response) is answer
    assert not outcomes


@pytest.mark.parametrize(
    "message",
    [
        AIMessage(content="Answer"),
        AIMessage(content=[{"type": "text", "text": "Answer"}]),
        AIMessage(
            content="", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}]
        ),
    ],
)
def test_valid_response_is_preserved(message):
    async def handler(request):
        return message

    response = asyncio.run(ModelRetryMiddleware().awrap_model_call(None, handler))

    assert response is message


@pytest.mark.parametrize("max_retries", [0, 2])
@pytest.mark.parametrize("reason", ["STOP", "MALFORMED_FUNCTION_CALL"])
def test_retryable_response_exhaustion_raises(max_retries, reason):
    calls = []
    response = ModelResponse(
        result=[
            HumanMessage(content="Question"),
            AIMessage(content="", response_metadata={"finish_reason": reason}),
        ]
    )

    async def handler(request):
        calls.append(request)
        return response

    with pytest.raises(
        MalformedResponseError, match="empty response" if reason == "STOP" else reason
    ):
        asyncio.run(
            ModelRetryMiddleware(
                max_retries=max_retries, initial_delay=0
            ).awrap_model_call(None, handler)
        )

    assert len(calls) == max_retries + 1


def test_finish_reason_comes_from_last_wrapped_ai_message():
    response = ModelResponse(
        result=[
            AIMessage(
                content="Earlier answer", response_metadata={"finish_reason": "STOP"}
            ),
            AIMessage(
                content="Malformed call",
                response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
            ),
            HumanMessage(content="Question"),
        ]
    )

    async def handler(request):
        return response

    with pytest.raises(MalformedResponseError, match="MALFORMED_FUNCTION_CALL"):
        asyncio.run(ModelRetryMiddleware(max_retries=0).awrap_model_call(None, handler))


def test_exception_retries_preserve_last_error():
    error = RuntimeError("Provider unavailable")
    calls = []

    async def handler(request):
        calls.append(request)
        raise error

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            ModelRetryMiddleware(initial_delay=0).awrap_model_call(None, handler)
        )

    assert caught.value is error
    assert len(calls) == 3


@pytest.mark.parametrize("failure_reason", [None, "length", "MAX_TOKENS"])
@pytest.mark.parametrize("successful_model", [0, 1, 2, None])
def test_agent_fallback_retries_each_model_before_advancing(
    monkeypatch, successful_model, failure_reason
):
    from agent import docs_agent_middleware
    from src.agent import config

    fallback = config.model_fallback_middleware
    retry = config.model_retry_middleware
    monkeypatch.setattr(retry, "initial_delay", 0)
    middleware = [
        m
        for m in docs_agent_middleware
        if isinstance(m, (ModelFallbackMiddleware, ModelRetryMiddleware))
    ]
    chain = _chain_async_model_call_handlers([m.awrap_model_call for m in middleware])
    models = [config.default_model, *fallback.models]
    calls = []

    async def handler(request):
        index = next(i for i, model in enumerate(models) if request.model is model)
        calls.append(index)
        return ModelResponse(
            result=[
                AIMessage(
                    content="Answer"
                    if index == successful_model
                    else ("Partial answer" if failure_reason else ""),
                    response_metadata={"finish_reason": failure_reason}
                    if index != successful_model and failure_reason
                    else {},
                )
            ]
        )

    request = ModelRequest(model=models[0], messages=[HumanMessage(content="Question")])
    if successful_model is None:
        with pytest.raises(
            MalformedResponseError, match=failure_reason or "empty response"
        ):
            asyncio.run(chain(request, handler))
        expected = [i for i in range(3) for _ in range(retry.max_retries + 1)]
    else:
        result = asyncio.run(chain(request, handler))
        assert result.model_response.result[-1].content == "Answer"
        expected = [
            i for i in range(successful_model) for _ in range(retry.max_retries + 1)
        ] + [successful_model]

    assert calls == expected
    assert [model.id for model in config.FALLBACK_MODELS] == [
        "openai:gpt-5.4-nano",
        "anthropic:claude-haiku-4-5-20251001",
    ]


def test_empty_response_uses_exponential_backoff(monkeypatch):
    delays = []
    outcomes = [
        AIMessage(content=""),
        AIMessage(content=" "),
        AIMessage(content="Answer"),
    ]

    async def sleep(delay):
        delays.append(delay)

    async def handler(request):
        return outcomes.pop(0)

    monkeypatch.setattr("src.middleware.retry_middleware.asyncio.sleep", sleep)
    result = asyncio.run(
        ModelRetryMiddleware(initial_delay=0.5, backoff_factor=2).awrap_model_call(
            None, handler
        )
    )

    assert result.content == "Answer"
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize("reason", ["length", "MAX_TOKENS"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_truncated_final_response_is_retried(reason, wrapped):
    outcomes = [
        AIMessage(
            content="Partial answer", response_metadata={"finish_reason": reason}
        ),
        AIMessage(content="Complete answer"),
    ]

    async def handler(request):
        message = outcomes.pop(0)
        return ModelResponse(result=[message]) if wrapped else message

    result = asyncio.run(
        ModelRetryMiddleware(initial_delay=0).awrap_model_call(None, handler)
    )
    assert (result.result[-1] if wrapped else result).content == "Complete answer"
    assert not outcomes


@pytest.mark.parametrize("reason", ["length", "MAX_TOKENS"])
def test_length_limited_tool_call_is_preserved(reason):
    message = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
        response_metadata={"finish_reason": reason},
    )

    async def handler(request):
        return message

    assert (
        asyncio.run(ModelRetryMiddleware().awrap_model_call(None, handler)) is message
    )


def test_stalled_attempt_hits_deadline_and_retries():
    answer = AIMessage(content="Here is an answer")
    calls = []

    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            await asyncio.sleep(10)
        return answer

    response = asyncio.run(
        ModelRetryMiddleware(initial_delay=0, attempt_deadline=0.05).awrap_model_call(
            None, handler
        )
    )

    assert response is answer
    assert len(calls) == 2


def test_stalled_attempts_raise_for_fallback_after_retries():
    from src.middleware.retry_middleware import ModelAttemptTimeoutError

    async def handler(request):
        await asyncio.sleep(10)

    with pytest.raises(ModelAttemptTimeoutError):
        asyncio.run(
            ModelRetryMiddleware(
                max_retries=1, initial_delay=0, attempt_deadline=0.05
            ).awrap_model_call(None, handler)
        )


@pytest.mark.parametrize(
    "model_name", ["gemini-3.5-flash-lite", "gemini-3-pro-preview"]
)
def test_prefilling_value_error_advances_to_fallback_once(model_name):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from src.middleware.model_fallback_middleware import (
        RepairingModelFallbackMiddleware,
    )

    primary = FakeListChatModel(responses=["Primary"])
    fallback_model = FakeListChatModel(responses=["Fallback"])
    fallback = RepairingModelFallbackMiddleware(fallback_model)
    retry = ModelRetryMiddleware(initial_delay=0)
    chain = _chain_async_model_call_handlers(
        [fallback.awrap_model_call, retry.awrap_model_call]
    )
    error = ValueError(
        f"Model '{model_name}' does not support model prefilling. The final "
        "request turn must be a user message or a function response."
    )
    messages = [HumanMessage(content="Question"), AIMessage(content="Previous answer")]
    request = ModelRequest(model=primary, messages=messages)
    calls = []
    answer = ModelResponse(result=[AIMessage(content="Fallback answer")])

    async def handler(request):
        calls.append(request.model)
        assert request.messages == messages
        if request.model is primary:
            raise error
        return answer

    result = asyncio.run(chain(request, handler))

    assert result.model_response is answer
    assert calls == [primary, fallback_model]
