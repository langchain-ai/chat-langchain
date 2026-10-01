"""Tests for primary model authentication failure handling."""

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage

from src.agent import config


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="hello")])


def test_google_api_key_invalid_uses_fallback_and_records_root_metadata(
    monkeypatch, caplog
):
    primary = object()
    fallback = object()
    middleware = config.PrimaryModelFallbackMiddleware.__new__(
        config.PrimaryModelFallbackMiddleware
    )
    middleware.models = [fallback]
    config.PrimaryModelFallbackMiddleware._auth_failure_logged = False
    metadata = {}
    monkeypatch.setattr(config, "set_run_metadata", metadata.update)
    calls = []

    def handler(request):
        calls.append(request.model)
        if request.model is primary:
            raise RuntimeError("400 API_KEY_INVALID")
        return "fallback answer"

    with caplog.at_level("ERROR", logger=config.logger.name):
        result = middleware.wrap_model_call(_request(primary), handler)

    assert result == "fallback answer"
    assert calls == [primary, fallback]
    assert metadata == {"primary_model_error": "auth", "served_by_fallback": True}
    assert (
        caplog.messages.count(
            "Primary model authentication failed for Gemini 3.5 Flash Lite (google); using fallback models"
        )
        == 1
    )


def test_unrelated_bad_request_is_not_marked_as_authentication_failure(monkeypatch):
    primary = object()
    middleware = config.PrimaryModelFallbackMiddleware.__new__(
        config.PrimaryModelFallbackMiddleware
    )
    middleware.models = []
    metadata = {}
    monkeypatch.setattr(config, "set_run_metadata", metadata.update)

    def handler(request):
        raise RuntimeError("400 INVALID_ARGUMENT")

    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        middleware.wrap_model_call(_request(primary), handler)

    assert metadata == {}


def test_missing_primary_key_logs_error(monkeypatch, caplog):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with caplog.at_level("ERROR"):
        config._log_primary_model_key_status()

    assert "GOOGLE_API_KEY is not configured for the primary model" in caplog.messages
