import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
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


def test_model_retry_wrapper_does_not_retry_400_invalid_request_error():
    class FooInvalidRequestError(Exception):
        status_code = 400

    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise FooInvalidRequestError("bad request")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(FooInvalidRequestError, match="bad request"):
        runnable.invoke("request")

    assert calls == 1


def test_model_retry_wrapper_does_not_retry_invalid_request_error_body():
    class ProviderError(Exception):
        body = {"error": {"type": "invalid_request_error"}}

    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise ProviderError("bad request")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(ProviderError, match="bad request"):
        runnable.invoke("request")

    assert calls == 1


def test_model_retry_wrapper_retries_generic_errors():
    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise RuntimeError("transient failure")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(RuntimeError, match="transient failure"):
        runnable.invoke("request")

    assert calls == 3


def test_model_retry_returns_message_for_deterministic_provider_failure():
    class ProviderError(Exception):
        http_status = 400

    calls = 0

    async def handler(_request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ProviderError("bad request")

    response = asyncio.run(
        ModelRetryMiddleware(max_retries=2, initial_delay=0).awrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert calls == 1
    assert len(response.result) == 1
    assert isinstance(response.result[0], AIMessage)
    assert response.result[0].content


def test_model_retry_retries_generic_errors_max_retries_plus_one_times():
    calls = 0

    async def handler(_request: ModelRequest):
        nonlocal calls
        calls += 1
        raise RuntimeError("transient failure")

    with pytest.raises(RuntimeError, match="transient failure"):
        asyncio.run(
            ModelRetryMiddleware(max_retries=2, initial_delay=0).awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 3
