import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
    _ProviderValidationAwareRunnableWithFallbacks,
    is_non_retryable_request_error,
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


def test_http_400_is_non_retryable_request_error():
    class BadRequestError(Exception):
        status_code = 400

    assert is_non_retryable_request_error(BadRequestError())


def test_model_fallback_stops_on_provider_validation_error():
    calls = []

    class BadRequestError(Exception):
        status_code = 400

    def primary(_input):
        calls.append("primary")
        raise BadRequestError("invalid request")

    def fallback(_input):
        calls.append("fallback")
        return "ok"

    runnable = _ProviderValidationAwareRunnableWithFallbacks(
        runnable=RunnableLambda(primary), fallbacks=[RunnableLambda(fallback)]
    )

    with pytest.raises(BadRequestError, match="invalid request"):
        runnable.invoke("request")

    assert calls == ["primary"]


def test_model_fallback_surfaces_first_transient_error():
    first_error = RuntimeError("primary unavailable")

    def primary(_input):
        raise first_error

    def fallback(_input):
        raise RuntimeError("fallback unavailable")

    runnable = _ProviderValidationAwareRunnableWithFallbacks(
        runnable=RunnableLambda(primary), fallbacks=[RunnableLambda(fallback)]
    )

    with pytest.raises(RuntimeError, match="primary unavailable") as error:
        runnable.invoke("request")

    assert error.value is first_error
