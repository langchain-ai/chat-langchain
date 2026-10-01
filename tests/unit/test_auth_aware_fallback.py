"""Tests for auth-aware model fallback behavior."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.runnables import Runnable

from src.agent import config
from src.utils.model_errors import is_auth_error


class FakeRunnable(Runnable):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)


@pytest.fixture(autouse=True)
def reset_breaker(monkeypatch):
    monkeypatch.setattr(config, "_auth_circuit_breaker_until", 0.0)


def test_auth_error_patterns():
    assert is_auth_error(RuntimeError("API_KEY_INVALID"))
    assert is_auth_error(RuntimeError("INVALID_ARGUMENT: API key not valid"))
    assert is_auth_error(RuntimeError("PERMISSION_DENIED"))

    error = RuntimeError("request failed")
    error.status_code = 401
    assert is_auth_error(error)
    assert not is_auth_error(RuntimeError("rate limit"))


def test_runnable_auth_failure_opens_breaker_and_bypasses_primary(monkeypatch):
    primary = FakeRunnable([RuntimeError("API_KEY_INVALID")])
    fallback = FakeRunnable(["fallback", "fallback"])
    model = config._AuthAwareRunnableFallback(
        primary, [("openai:gpt-5.4-nano", fallback)]
    )

    assert model.invoke("input") == "fallback"
    assert model.invoke("input") == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 2


def test_runnable_transient_failure_uses_fallback_without_breaker():
    primary = FakeRunnable([RuntimeError("timeout")])
    fallback = FakeRunnable(["fallback"])
    model = config._AuthAwareRunnableFallback(
        primary, [("openai:gpt-5.4-nano", fallback)]
    )

    assert model.invoke("input") == "fallback"
    assert config._auth_circuit_breaker_until == 0.0


def test_runnable_records_fallback_metadata(monkeypatch):
    metadata = {}

    class FakeRunTree:
        def add_metadata(self, values):
            metadata.update(values)

    monkeypatch.setattr(config.ls, "get_current_run_tree", lambda: FakeRunTree())
    primary = FakeRunnable([RuntimeError("API_KEY_INVALID")])
    fallback = FakeRunnable(["fallback"])
    model = config._AuthAwareRunnableFallback(
        primary, [("openai:gpt-5.4-nano", fallback)]
    )

    model.invoke("input")

    assert metadata == {"served_by_fallback_model": "openai:gpt-5.4-nano"}


def test_breaker_expires_after_300_seconds(monkeypatch):
    now = 100.0
    monkeypatch.setattr(config.time, "monotonic", lambda: now)
    config._open_auth_circuit()
    assert config._auth_circuit_is_open()

    monkeypatch.setattr(config.time, "monotonic", lambda: now + 300.0)
    assert not config._auth_circuit_is_open()


def test_async_auth_failure_bypasses_primary():
    primary = FakeRunnable([RuntimeError("API_KEY_INVALID")])
    fallback = FakeRunnable(["fallback", "fallback"])
    model = config._AuthAwareRunnableFallback(
        primary, [("openai:gpt-5.4-nano", fallback)]
    )

    assert asyncio.run(model.ainvoke("input")) == "fallback"
    assert asyncio.run(model.ainvoke("input")) == "fallback"
    assert primary.calls == 1


def test_agent_middleware_auth_failure_bypasses_primary():
    middleware = config.AuthAwareModelFallbackMiddleware.__new__(
        config.AuthAwareModelFallbackMiddleware
    )
    middleware.models = [object()]
    middleware.fallback_model_ids = ("openai:gpt-5.4-nano",)
    primary_model = object()
    request = ModelRequest(model=primary_model, messages=[])
    calls = []

    def handler(model_request):
        calls.append(model_request.model)
        if len(calls) == 1:
            raise RuntimeError("API_KEY_INVALID")
        return "fallback"

    assert middleware.wrap_model_call(request, handler) == "fallback"
    assert calls == [primary_model, middleware.models[0]]
