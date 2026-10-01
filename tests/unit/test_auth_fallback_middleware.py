import asyncio

import langsmith as ls
from langchain.agents.middleware import model_fallback
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.auth_fallback_middleware import (
    AuthenticationAwareModelFallbackMiddleware,
)


class FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FakeRun:
    def __init__(self):
        self.metadata = {}
        self.tags = []


def _request() -> ModelRequest:
    return ModelRequest(
        model=FakeModel("primary"), messages=[HumanMessage(content="Hi")]
    )


def _response(text: str) -> ModelResponse:
    return ModelResponse(result=[AIMessage(content=text)])


def _middleware(monkeypatch):
    monkeypatch.setattr(model_fallback, "init_chat_model", FakeModel)
    return AuthenticationAwareModelFallbackMiddleware(
        "primary", "fallback", cooldown_seconds=300
    )


def test_google_auth_error_uses_fallback_and_cooldown(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(ls, "get_current_run_tree", lambda: run)
    middleware = _middleware(monkeypatch)
    calls = []

    async def handler(request):
        calls.append(request.model.model_name)
        if request.model.model_name == "primary":
            raise ProviderError("API_KEY_INVALID")
        return _response("fallback answer")

    first = asyncio.run(middleware.awrap_model_call(_request(), handler))
    second = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert first.result[0].content == "fallback answer"
    assert second.result[0].content == "fallback answer"
    assert calls == ["primary", "fallback", "fallback"]
    assert run.metadata == {
        "served_by_fallback": True,
        "primary_model_error": "ProviderError: API_KEY_INVALID",
    }
    assert run.tags == ["served_by_fallback"]


def test_401_auth_error_is_recorded(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(ls, "get_current_run_tree", lambda: run)
    middleware = _middleware(monkeypatch)

    async def handler(request):
        if request.model.model_name == "primary":
            raise ProviderError("unauthorized", status_code=401)
        return _response("fallback answer")

    asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert "401" in run.metadata["primary_model_error"]


def test_transient_error_does_not_start_cooldown(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(ls, "get_current_run_tree", lambda: run)
    middleware = _middleware(monkeypatch)
    primary_calls = 0

    async def handler(request):
        nonlocal primary_calls
        if request.model.model_name == "primary":
            primary_calls += 1
            raise ProviderError("temporary outage", status_code=500)
        return _response("fallback answer")

    asyncio.run(middleware.awrap_model_call(_request(), handler))
    asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert primary_calls == 2
    assert run.metadata == {}
