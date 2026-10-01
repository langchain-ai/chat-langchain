import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware import primary_auth_breaker_middleware as middleware_module
from src.middleware.primary_auth_breaker_middleware import (
    PrimaryAuthBreaker,
    PrimaryAuthBreakerMiddleware,
)


class FakeModel:
    def __init__(self, model_id: str):
        self.model_id = model_id


class AuthError(Exception):
    status_code = 401


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="Hi")])


def _response():
    return ModelResponse(result=[AIMessage(content="ok")])


def test_auth_failure_trips_breaker_and_skips_primary(monkeypatch):
    models = {model_id: FakeModel(model_id) for model_id in ("primary", "fallback")}
    monkeypatch.setattr(
        middleware_module,
        "init_chat_model",
        lambda model_id: models[model_id],
    )
    metadata = {}
    monkeypatch.setattr(middleware_module, "_root_run_metadata", lambda: metadata)
    middleware = PrimaryAuthBreakerMiddleware(
        "primary", ["fallback"], breaker=PrimaryAuthBreaker(ttl=60)
    )
    calls = []

    def handler(request):
        calls.append(request.model.model_id)
        if request.model.model_id == "primary":
            raise RuntimeError("API_KEY_INVALID")
        return _response()

    middleware.wrap_model_call(_request(models["primary"]), handler)
    middleware.wrap_model_call(_request(models["primary"]), handler)

    assert calls == ["primary", "fallback", "fallback"]
    assert metadata == {
        "served_by_fallback": True,
        "primary_error_kind": "auth",
        "ls_served_model": "fallback",
    }


def test_transient_failure_falls_back_without_opening_breaker(monkeypatch):
    models = {model_id: FakeModel(model_id) for model_id in ("primary", "fallback")}
    monkeypatch.setattr(
        middleware_module,
        "init_chat_model",
        lambda model_id: models[model_id],
    )
    middleware = PrimaryAuthBreakerMiddleware("primary", ["fallback"])
    calls = []

    def handler(request):
        calls.append(request.model.model_id)
        if request.model.model_id == "primary":
            raise TimeoutError("timed out")
        return _response()

    middleware.wrap_model_call(_request(models["primary"]), handler)

    assert calls == ["primary", "fallback"]
    assert not middleware.breaker.is_open()


def test_async_fallback_sets_root_metadata(monkeypatch):
    models = {model_id: FakeModel(model_id) for model_id in ("primary", "fallback")}
    monkeypatch.setattr(
        middleware_module,
        "init_chat_model",
        lambda model_id: models[model_id],
    )
    metadata = {}
    monkeypatch.setattr(middleware_module, "_root_run_metadata", lambda: metadata)
    middleware = PrimaryAuthBreakerMiddleware("primary", ["fallback"])

    async def handler(request):
        if request.model.model_id == "primary":
            raise AuthError("unauthorized")
        return _response()

    asyncio.run(middleware.awrap_model_call(_request(models["primary"]), handler))

    assert metadata["served_by_fallback"] is True
    assert metadata["primary_error_kind"] == "auth"
    assert metadata["ls_served_model"] == "fallback"
