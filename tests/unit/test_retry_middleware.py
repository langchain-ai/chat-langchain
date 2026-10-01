import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

import src.middleware.retry_middleware as retry_module
from src.middleware.retry_middleware import (
    AuthAwareModelFallbackMiddleware,
    ModelRetryMiddleware,
    PrimaryModelCircuitBreaker,
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


class AuthenticationError(Exception):
    pass


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="Hi")])


def test_auth_error_trips_breaker_and_skips_primary_on_next_call():
    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        circuit_breaker=PrimaryModelCircuitBreaker(),
    )
    primary = SimpleNamespace(model="gemini-3.5-flash-lite")
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        raise AuthenticationError("invalid credentials")

    with pytest.raises(AuthenticationError):
        asyncio.run(middleware.awrap_model_call(_request(primary), handler))
    with pytest.raises(RuntimeError, match="circuit breaker"):
        asyncio.run(middleware.awrap_model_call(_request(primary), handler))

    assert calls == 1
    assert middleware.circuit_breaker.auth_error_count == 1


def test_fallback_served_after_auth_error_has_metadata(monkeypatch):
    circuit_breaker = PrimaryModelCircuitBreaker()
    retry = ModelRetryMiddleware(
        max_retries=0,
        circuit_breaker=circuit_breaker,
    )
    fallback_model = SimpleNamespace(model="gpt-5.4-nano")
    fallback = AuthAwareModelFallbackMiddleware(
        fallback_model,
        circuit_breaker=circuit_breaker,
    )
    config = {"metadata": {}}
    monkeypatch.setattr(retry_module, "get_config", lambda: config)
    primary = SimpleNamespace(model="gemini-3.5-flash-lite")
    calls = []

    async def underlying(request):
        calls.append(request.model)
        if request.model is primary:
            raise AuthenticationError("API_KEY_INVALID")
        return ModelResponse(result=[])

    async def handler(request):
        return await retry.awrap_model_call(request, underlying)

    result = asyncio.run(fallback.awrap_model_call(_request(primary), handler))

    assert isinstance(result, ModelResponse)
    assert calls == [primary, fallback_model]
    assert config["metadata"] == {
        "served_by_fallback": True,
        "primary_model_error": "auth",
    }


def test_transient_error_retries_without_tripping_breaker():
    circuit_breaker = PrimaryModelCircuitBreaker()
    middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        circuit_breaker=circuit_breaker,
    )
    primary = SimpleNamespace(model="gemini-3.5-flash-lite")
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary failure")
        return ModelResponse(result=[])

    result = asyncio.run(middleware.awrap_model_call(_request(primary), handler))

    assert isinstance(result, ModelResponse)
    assert calls == 2
    assert not circuit_breaker.open
