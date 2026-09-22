import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)


class _ProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"provider error {status_code}")
        self.status_code = status_code


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


def test_http_400_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise _ProviderError(400)

    with pytest.raises(_ProviderError, match="provider error 400"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_http_429_error_is_retried():
    max_retries = 2
    middleware = ModelRetryMiddleware(max_retries=max_retries, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise _ProviderError(429)

    with pytest.raises(_ProviderError, match="provider error 429"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == max_retries + 1


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
