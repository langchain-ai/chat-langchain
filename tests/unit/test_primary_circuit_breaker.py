import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

import src.middleware.primary_circuit_breaker as breaker_module
from src.middleware.primary_circuit_breaker import PrimaryModelCircuitBreaker


class AuthenticationError(Exception):
    pass


class TransientError(Exception):
    pass


class FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __str__(self) -> str:
        return self.model_name


def request(model):
    return ModelRequest(model=model, messages=[])


def response():
    return ModelResponse(result=[AIMessage(content="ok")])


def test_authentication_error_opens_breaker_and_bypasses_primary(monkeypatch):
    now = [0.0]
    primary = FakeModel("gemini-3.5-flash-lite")
    fallback = FakeModel("gpt-5.4-nano")
    middleware = PrimaryModelCircuitBreaker(
        "google_genai:gemini-3.5-flash-lite", fallback, clock=lambda: now[0]
    )
    calls = []

    def handler(model_request):
        calls.append(model_request.model)
        if model_request.model is primary:
            raise AuthenticationError("API_KEY_INVALID")
        return response()

    try:
        middleware.wrap_model_call(request(primary), handler)
    except AuthenticationError:
        pass

    assert middleware.is_open
    assert middleware.wrap_model_call(request(primary), handler).result
    assert calls == [primary, fallback]


def test_transient_error_does_not_open_breaker():
    primary = FakeModel("gemini-3.5-flash-lite")
    fallback = FakeModel("gpt-5.4-nano")
    middleware = PrimaryModelCircuitBreaker(
        "google_genai:gemini-3.5-flash-lite", fallback
    )

    def handler(_request):
        raise TransientError("HTTP 503")

    try:
        middleware.wrap_model_call(request(primary), handler)
    except TransientError:
        pass

    assert not middleware.is_open


def test_cooldown_resets_breaker():
    now = [0.0]
    fallback = FakeModel("gpt-5.4-nano")
    middleware = PrimaryModelCircuitBreaker(
        "google_genai:gemini-3.5-flash-lite",
        fallback,
        cooldown_seconds=5,
        clock=lambda: now[0],
    )
    middleware.open(AuthenticationError("permission denied"))
    assert middleware.is_open
    now[0] = 5
    assert not middleware.is_open


def test_async_fallback_records_root_run_metadata(monkeypatch):
    metadata = {}
    monkeypatch.setattr(
        breaker_module,
        "set_run_metadata",
        lambda **values: metadata.update(values),
    )
    primary = FakeModel("gemini-3.5-flash-lite")
    fallback = FakeModel("gpt-5.4-nano")
    middleware = PrimaryModelCircuitBreaker(
        "google_genai:gemini-3.5-flash-lite", fallback
    )

    async def handler(model_request):
        if model_request.model is primary:
            raise AuthenticationError("API_KEY_INVALID")
        return response()

    try:
        asyncio.run(middleware.awrap_model_call(request(primary), handler))
    except AuthenticationError:
        asyncio.run(middleware.awrap_model_call(request(primary), handler))

    assert metadata == {
        "served_by_fallback": True,
        "primary_error_class": "AuthenticationError",
        "primary_error_reason": "API_KEY_INVALID",
    }
