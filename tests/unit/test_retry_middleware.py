import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
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


def test_authentication_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise RuntimeError("API_KEY_INVALID")

    with pytest.raises(RuntimeError, match="API_KEY_INVALID"):
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


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="Hi")])


def test_auth_failure_logs_once_and_bypasses_primary_during_cooldown(caplog, monkeypatch):
    breaker = PrimaryModelCircuitBreaker(cooldown_seconds=60)
    middleware = AuthAwareModelFallbackMiddleware(object(), breaker=breaker)
    primary = object()
    primary_calls = 0
    fallback_calls = 0
    run_tree = type("RunTree", (), {"metadata": {}, "tags": []})()
    monkeypatch.setattr(retry_module.ls, "get_current_run_tree", lambda: run_tree)

    async def handler(request):
        nonlocal primary_calls, fallback_calls
        if request.model is primary:
            primary_calls += 1
            raise RuntimeError("API_KEY_INVALID")
        fallback_calls += 1
        return AIMessage(content="fallback")

    request = _request(primary)
    with caplog.at_level("ERROR"):
        first = asyncio.run(middleware.awrap_model_call(request, handler))
        second = asyncio.run(middleware.awrap_model_call(request, handler))

    assert first.content == "fallback"
    assert second.content == "fallback"
    assert primary_calls == 1
    assert fallback_calls == 2
    assert sum("authentication failure" in record.message for record in caplog.records) == 1
    assert run_tree.metadata == {
        "primary_model_unavailable": True,
        "fallback_reason": "auth_error",
    }
    assert run_tree.tags == ["primary_model_unavailable"]


def test_auth_circuit_allows_primary_after_cooldown():
    breaker = PrimaryModelCircuitBreaker(cooldown_seconds=60)
    middleware = AuthAwareModelFallbackMiddleware(object(), breaker=breaker)
    primary = object()
    primary_calls = 0

    async def handler(request):
        nonlocal primary_calls
        if request.model is primary:
            primary_calls += 1
            if primary_calls == 1:
                raise RuntimeError("API_KEY_INVALID")
        return AIMessage(content="response")

    request = _request(primary)
    asyncio.run(middleware.awrap_model_call(request, handler))
    breaker._opened_until = 0
    asyncio.run(middleware.awrap_model_call(request, handler))

    assert primary_calls == 2


def test_transient_failure_still_attempts_primary_each_time():
    breaker = PrimaryModelCircuitBreaker(cooldown_seconds=60)
    middleware = AuthAwareModelFallbackMiddleware(object(), breaker=breaker)
    primary = object()
    primary_calls = 0
    fallback_calls = 0

    async def handler(request):
        nonlocal primary_calls, fallback_calls
        if request.model is primary:
            primary_calls += 1
            raise TimeoutError("temporary outage")
        fallback_calls += 1
        return AIMessage(content="fallback")

    request = _request(primary)
    asyncio.run(middleware.awrap_model_call(request, handler))
    asyncio.run(middleware.awrap_model_call(request, handler))

    assert primary_calls == 2
    assert fallback_calls == 2
