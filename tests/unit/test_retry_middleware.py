import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)


def test_provider_invalid_request_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    class AnthropicInvalidRequestError(Exception):
        status_code = 400

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise AnthropicInvalidRequestError("unsupported request shape")

    response = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert calls == 1
    assert isinstance(response.result[0], AIMessage)


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


def test_transient_error_is_retried_max_attempts():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise RuntimeError("temporary failure")

    response = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert calls == middleware.max_retries + 1
    assert isinstance(response.result[0], AIMessage)
    assert response.result[0].content


def test_exhausted_call_returns_non_empty_ai_message():
    middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)

    async def handler(request: ModelRequest):
        raise RuntimeError("temporary failure")

    response = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert isinstance(response.result[0], AIMessage)
    assert response.result[0].content
