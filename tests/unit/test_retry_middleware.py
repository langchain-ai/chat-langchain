import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)


class ProviderError(Exception):
    def __init__(self, status_code):
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


def test_provider_400_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ProviderError(400)

    with pytest.raises(ProviderError, match="provider error 400"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


@pytest.mark.parametrize("status_code", [408, 429, 503])
def test_retryable_provider_status_is_retried(status_code):
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ProviderError(status_code)

    with pytest.raises(ProviderError, match=f"provider error {status_code}"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == middleware.max_retries + 1


def test_provider_response_400_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        error = Exception("provider error")
        error.response = SimpleNamespace(status_code=400)
        raise error

    with pytest.raises(Exception, match="provider error"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_malformed_function_call_is_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"}
        )

    response = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert calls == middleware.max_retries + 1
    assert response.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL"


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


def test_model_retry_wrapper_does_not_retry_provider_400_error():
    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise ProviderError(400)

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(ProviderError, match="provider error 400"):
        runnable.invoke("request")

    assert calls == 1
