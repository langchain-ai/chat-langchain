import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
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


def test_model_retry_middleware_does_not_retry_authentication_error(monkeypatch):
    from src.middleware.retry_middleware import ProviderAuthenticationError

    monkeypatch.setenv("MODEL_AUTH_VALIDATION_STRICT", "true")
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise RuntimeError("API_KEY_INVALID")

    with pytest.raises(ProviderAuthenticationError):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_model_fallback_middleware_propagates_authentication_error(monkeypatch):
    from src.middleware.retry_middleware import (
        ProviderAuthenticationError,
        ProviderAwareModelFallbackMiddleware,
    )

    monkeypatch.setenv("MODEL_AUTH_VALIDATION_STRICT", "true")
    middleware = ProviderAwareModelFallbackMiddleware("openai:gpt-5.4-nano")
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise RuntimeError("API_KEY_INVALID")

    with pytest.raises(ProviderAuthenticationError):
        middleware.wrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )

    assert calls == 1


def test_non_strict_fallback_records_authentication_failure(monkeypatch):
    from src.middleware.retry_middleware import ProviderAwareModelFallbackMiddleware

    monkeypatch.setenv("MODEL_AUTH_VALIDATION_STRICT", "false")
    middleware = ProviderAwareModelFallbackMiddleware("openai:gpt-5.4-nano")
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise RuntimeError("API_KEY_INVALID")
        return "fallback response"

    result = middleware.wrap_model_call(
        ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
        handler,
    )

    assert result == "fallback response"
    assert requests[1].state["fallback_service"]
    assert requests[1].state["primary_auth_failure_reason"] == "API_KEY_INVALID"
