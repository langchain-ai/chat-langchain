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


def test_auth_error_classification():
    from src.middleware.retry_middleware import is_permanent_auth_failure

    class ResponseError(Exception):
        status_code = 403

    class GoogleResponseError(Exception):
        code = 401

    assert is_permanent_auth_failure(ValueError("API_KEY_INVALID"))
    assert is_permanent_auth_failure(ValueError("API key not valid"))
    assert is_permanent_auth_failure(ResponseError())
    assert is_permanent_auth_failure(GoogleResponseError())
    assert not is_permanent_auth_failure(ValueError("rate limited"))


def test_auth_failure_opens_circuit_and_uses_first_fallback(monkeypatch):
    from src.middleware.retry_middleware import ModelAuthFallbackMiddleware

    class FakeModel:
        def __init__(self, model_id):
            self.model = model_id

    monkeypatch.setattr(
        "langchain.agents.middleware.model_fallback.init_chat_model", FakeModel
    )
    ModelAuthFallbackMiddleware.reset_circuit()
    middleware = ModelAuthFallbackMiddleware("primary", "fallback", "second")
    calls = []

    def handler(request):
        calls.append(request.model.model)
        if request.model.model == "primary":
            raise RuntimeError("API_KEY_INVALID")
        return "ok"

    result = middleware.wrap_model_call(
        ModelRequest(model=FakeModel("primary"), messages=[HumanMessage(content="Hi")]),
        handler,
    )

    assert result == "ok"
    assert calls == ["primary", "fallback"]
    assert middleware._circuit_is_open(60)


def test_open_auth_circuit_skips_primary_and_records_metadata(monkeypatch):
    from src.middleware.retry_middleware import ModelAuthFallbackMiddleware

    class FakeModel:
        def __init__(self, model_id):
            self.model = model_id

    monkeypatch.setattr(
        "langchain.agents.middleware.model_fallback.init_chat_model", FakeModel
    )
    metadata = {}
    monkeypatch.setattr(
        "langgraph.config.get_config", lambda: {"metadata": metadata}
    )
    ModelAuthFallbackMiddleware.reset_circuit()
    middleware = ModelAuthFallbackMiddleware("primary", "fallback", cooldown_seconds=60)
    middleware._open_circuit(RuntimeError("API_KEY_INVALID"))
    calls = []

    async def handler(request):
        calls.append(request.model.model)
        return "ok"

    result = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=FakeModel("primary"), messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert result == "ok"
    assert calls == ["fallback"]
    assert metadata == {
        "served_by_model": "fallback",
        "fallback_reason": "primary_auth_circuit_open",
    }


def test_auth_circuit_cooldown_allows_primary_and_resets(monkeypatch):
    import src.middleware.retry_middleware as retry_module
    from src.middleware.retry_middleware import ModelAuthFallbackMiddleware

    class FakeModel:
        def __init__(self, model_id):
            self.model = model_id

    monkeypatch.setattr(
        "langchain.agents.middleware.model_fallback.init_chat_model", FakeModel
    )
    ModelAuthFallbackMiddleware.reset_circuit()
    middleware = ModelAuthFallbackMiddleware("primary", "fallback", cooldown_seconds=1)
    ModelAuthFallbackMiddleware._circuit_opened_at = 0
    monkeypatch.setattr(retry_module.time, "monotonic", lambda: 2)
    calls = []

    def handler(request):
        calls.append(request.model.model)
        return "primary-ok"

    assert (
        middleware.wrap_model_call(
            ModelRequest(model=FakeModel("primary"), messages=[HumanMessage(content="Hi")]),
            handler,
        )
        == "primary-ok"
    )
    assert calls == ["primary"]
    assert not middleware._circuit_is_open(1)
