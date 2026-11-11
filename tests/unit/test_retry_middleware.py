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


@pytest.mark.parametrize("successful_model", [0, 1, 2, None])
def test_agent_fallback_retries_each_model_before_advancing(
    monkeypatch, successful_model
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
            result=[AIMessage(content="Answer" if index == successful_model else "")]
        )

    request = ModelRequest(model=models[0], messages=[HumanMessage(content="Question")])
    if successful_model is None:
        with pytest.raises(MalformedResponseError, match="empty response"):
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
