import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    AuthAwareModelFallbackMiddleware,
    ModelRetryMiddleware,
    ProviderCircuitOpenError,
    _ProviderValidationAwareRunnableRetry,
    provider_is_available,
    reset_provider_breakers,
    trip_provider,
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
    status_code = 401


class TransientError(Exception):
    status_code = 503


def test_auth_error_trips_breaker_and_skips_later_calls():
    reset_provider_breakers()
    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise AuthenticationError("invalid credentials")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3, provider="google"
    )
    with pytest.raises(AuthenticationError):
        runnable.invoke("request")
    with pytest.raises(ProviderCircuitOpenError):
        runnable.invoke("request")
    assert calls == 1


def test_transient_error_does_not_trip_breaker():
    reset_provider_breakers()

    def invoke(_input):
        raise TransientError("overloaded")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=1, provider="google"
    )
    with pytest.raises(TransientError):
        runnable.invoke("request")
    assert provider_is_available("google")


def test_breaker_resets_after_cooldown(monkeypatch):
    reset_provider_breakers()
    clock = iter((100.0, 100.0, 401.0))
    monkeypatch.setattr("src.middleware.retry_middleware.time.monotonic", lambda: next(clock))
    trip_provider("google")
    assert not provider_is_available("google")
    assert provider_is_available("google")


def test_fallback_records_served_model(monkeypatch):
    reset_provider_breakers()
    config = {"metadata": {}}
    monkeypatch.setattr("langgraph.config.get_config", lambda: config)

    class FakeModel:
        def __init__(self, model_name, provider):
            self.model_name = model_name
            self._llm_type = provider

    middleware = AuthAwareModelFallbackMiddleware.__new__(AuthAwareModelFallbackMiddleware)
    middleware.models = [FakeModel("gpt-5.4-nano", "openai")]
    request = ModelRequest(model=FakeModel("gemini-3.5-flash-lite", "google"), messages=[])

    def handler(current_request):
        if current_request.model._llm_type == "google":
            raise AuthenticationError("API_KEY_INVALID")
        return ModelResponse(result=[])

    result = middleware.wrap_model_call(request, handler)
    assert result.result == []
    assert config["metadata"]["served_model"] == "gpt-5.4-nano"
