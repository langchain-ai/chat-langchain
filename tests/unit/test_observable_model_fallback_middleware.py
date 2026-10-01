import asyncio
import logging

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage

from src.middleware.observable_model_fallback_middleware import (
    ObservableModelFallbackMiddleware,
)


class FakeModel:
    model_name = "fake-primary"


class AuthError(Exception):
    status_code = 403


def _request():
    return ModelRequest(model=FakeModel(), messages=[HumanMessage(content="hello")])


def _middleware(monkeypatch):
    middleware = ObservableModelFallbackMiddleware.__new__(
        ObservableModelFallbackMiddleware
    )
    middleware.models = [FakeModel()]
    monkeypatch.setattr(
        "src.middleware.observable_model_fallback_middleware._sanitize_request_for_fallback",
        lambda request, model: request,
        raising=False,
    )
    return middleware


def test_sync_auth_failure_falls_back_and_records_metadata(monkeypatch, caplog):
    middleware = _middleware(monkeypatch)
    run_tree = type("RunTree", (), {"metadata": {}})()
    monkeypatch.setattr(
        "src.middleware.observable_model_fallback_middleware.ls.get_current_run_tree",
        lambda: run_tree,
    )
    calls = []

    def handler(request):
        calls.append(request.model)
        if len(calls) == 1:
            raise RuntimeError("API_KEY_INVALID")
        return "fallback answer"

    with caplog.at_level(logging.ERROR):
        result = middleware.wrap_model_call(_request(), handler)

    assert result == "fallback answer"
    assert len(calls) == 2
    assert run_tree.metadata["served_by_fallback"] is True
    assert run_tree.metadata["primary_model_error"] == "API_KEY_INVALID"
    assert "failed authentication" in caplog.text


def test_async_auth_failure_falls_back_and_records_metadata(monkeypatch):
    middleware = _middleware(monkeypatch)
    run_tree = type("RunTree", (), {"metadata": {}})()
    monkeypatch.setattr(
        "src.middleware.observable_model_fallback_middleware.ls.get_current_run_tree",
        lambda: run_tree,
    )
    calls = []

    async def handler(request):
        calls.append(request.model)
        if len(calls) == 1:
            raise AuthError("forbidden")
        return "fallback answer"

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert result == "fallback answer"
    assert len(calls) == 2
    assert run_tree.metadata["served_by_fallback"] is True


def test_non_auth_failure_falls_back_without_auth_metadata(monkeypatch):
    middleware = _middleware(monkeypatch)
    run_tree = type("RunTree", (), {"metadata": {}})()
    monkeypatch.setattr(
        "src.middleware.observable_model_fallback_middleware.ls.get_current_run_tree",
        lambda: run_tree,
    )

    calls = []

    def handler(request):
        calls.append(request.model)
        if len(calls) == 1:
            raise RuntimeError("temporary outage")
        return "fallback answer"

    assert middleware.wrap_model_call(_request(), handler) == "fallback answer"
    assert "served_by_fallback" not in run_tree.metadata
