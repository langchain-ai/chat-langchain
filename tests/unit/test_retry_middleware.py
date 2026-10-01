import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import src.middleware.retry_middleware as retry_module
from src.middleware.retry_middleware import (
    ModelFallbackWithCircuitBreakerMiddleware,
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
    model_id,
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


def test_model_id_preserves_provider_for_initialized_models():
    model = SimpleNamespace(
        model="gemini-3.5-flash-lite", _llm_type="chat-google-generative-ai"
    )

    assert model_id(model) == "google_genai:gemini-3.5-flash-lite"


def test_authentication_failure_skips_primary_on_later_steps(caplog, monkeypatch):
    retry_module._AUTHENTICATION_FAILURE_MODELS.clear()
    primary = SimpleNamespace(model="primary")
    fallback = SimpleNamespace(model="fallback")
    middleware = ModelFallbackWithCircuitBreakerMiddleware.__new__(
        ModelFallbackWithCircuitBreakerMiddleware
    )
    middleware.models = [fallback]
    calls = {"primary": 0, "fallback": 0}
    outcomes = []
    monkeypatch.setattr(
        retry_module,
        "record_model_outcome",
        lambda model, *, fallback_used: outcomes.append((model.model, fallback_used)),
    )

    async def handler(request):
        calls[request.model.model] += 1
        if request.model is primary:
            error = RuntimeError("unauthorized")
            error.status_code = 401
            raise error
        return retry_module.ModelResponse([AIMessage(content="ok")])

    async def run():
        request = ModelRequest(model=primary, messages=[HumanMessage(content="Hi")])
        first = await middleware.awrap_model_call(request, handler)
        second = await middleware.awrap_model_call(request, handler)
        return first, second

    asyncio.run(run())

    assert calls == {"primary": 1, "fallback": 2}
    assert outcomes == [("fallback", True), ("fallback", True)]
    assert "primary" in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)
    retry_module._AUTHENTICATION_FAILURE_MODELS.clear()


def test_transient_failure_still_retries():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        error = RuntimeError("rate limited")
        error.status_code = 429
        raise error

    with pytest.raises(RuntimeError, match="rate limited"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=SimpleNamespace(model="primary"), messages=[]),
                handler,
            )
        )

    assert calls == 3
