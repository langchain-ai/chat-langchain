import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
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


def test_model_call_timeout_retries_then_propagates():
    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout_seconds=0.01,
    )
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 3


def test_model_call_timeout_uses_fallback_model(monkeypatch):
    primary_model = object()
    fallback_model = object()
    retry_middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        timeout_seconds=0.01,
    )
    monkeypatch.setattr(
        "langchain.agents.middleware.model_fallback.init_chat_model",
        lambda model: fallback_model,
    )
    fallback_middleware = ModelFallbackMiddleware("openai:gpt-4o-mini")
    fallback_middleware.models = [fallback_model]
    primary_calls = 0
    fallback_calls = 0

    async def handler(request: ModelRequest):
        nonlocal primary_calls, fallback_calls
        if request.model is primary_model:
            primary_calls += 1
            await asyncio.sleep(1)
        fallback_calls += 1
        return ModelResponse(result=[AIMessage(content="fallback")])

    async def retry_handler(request: ModelRequest):
        if request.model is primary_model:
            return await retry_middleware.awrap_model_call(request, handler)
        return await handler(request)

    result = asyncio.run(
        fallback_middleware.awrap_model_call(
            ModelRequest(model=primary_model, messages=[HumanMessage(content="Hi")]),
            retry_handler,
        )
    )

    assert result.result[0].content == "fallback"
    assert primary_calls == 2
    assert fallback_calls == 1


def test_sync_model_call_does_not_use_async_timeout():
    middleware = ModelRetryMiddleware(timeout_seconds=0.01)
    calls = 0

    def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="sync")])

    with pytest.raises(NotImplementedError):
        middleware.wrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )

    assert calls == 0
