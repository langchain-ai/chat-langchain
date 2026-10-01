import logging
import os

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from src.agent import config
from src.middleware import model_fallback_middleware as fallback_module
from src.middleware.model_fallback_middleware import ObservableModelFallbackMiddleware


class FakeModel:
    def __init__(self, name):
        self.model_name = name


class ApiKeyError(Exception):
    status_code = 400


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="Hello")])


def test_startup_credential_check_logs_auth_failure(monkeypatch, caplog):
    class FailingModel:
        def invoke(self, _prompt):
            raise ApiKeyError("400 INVALID_ARGUMENT API_KEY_INVALID")

    monkeypatch.setenv("VALIDATE_MODEL_CREDENTIALS", "true")
    monkeypatch.delenv("FAIL_ON_INVALID_PRIMARY_KEY", raising=False)
    monkeypatch.setattr(config, "init_chat_model", lambda **_kwargs: FailingModel())

    with caplog.at_level(logging.ERROR, logger=config.logger.name):
        config.validate_model_credentials()

    assert "Credential rejected for google model" in caplog.text
    assert "API_KEY_INVALID" in caplog.text


def test_startup_credential_check_can_fail_strictly(monkeypatch):
    class FailingModel:
        def invoke(self, _prompt):
            raise ApiKeyError("400 INVALID_ARGUMENT API_KEY_INVALID")

    monkeypatch.setenv("VALIDATE_MODEL_CREDENTIALS", "true")
    monkeypatch.setenv("FAIL_ON_INVALID_PRIMARY_KEY", "true")
    monkeypatch.setattr(config, "init_chat_model", lambda **_kwargs: FailingModel())

    with pytest.raises(ApiKeyError):
        config.validate_model_credentials()


def test_auth_fallback_records_metadata_and_opens_breaker(monkeypatch):
    ObservableModelFallbackMiddleware.reset_primary_breaker()
    run_tree = type("RunTree", (), {"metadata": {}})()
    monkeypatch.setattr(fallback_module, "get_current_run_tree", lambda: run_tree)
    primary = FakeModel("google_genai:gemini-3.5-flash-lite")
    fallback = FakeModel("openai:gpt-5.4-nano")
    middleware = ObservableModelFallbackMiddleware(fallback)
    calls = []

    def handler(request):
        calls.append(request.model.model_name)
        if request.model is primary:
            raise ApiKeyError("API_KEY_INVALID")
        return "fallback response"

    assert middleware.wrap_model_call(_request(primary), handler) == "fallback response"
    assert run_tree.metadata == {
        "served_by_fallback": True,
        "fallback_reason": "API_KEY_INVALID",
        "served_model": "openai:gpt-5.4-nano",
    }
    calls.clear()
    assert middleware.wrap_model_call(_request(primary), handler) == "fallback response"
    assert calls == ["openai:gpt-5.4-nano"]
    ObservableModelFallbackMiddleware.reset_primary_breaker()


def test_transient_fallback_does_not_open_breaker():
    ObservableModelFallbackMiddleware.reset_primary_breaker()
    primary = FakeModel("google_genai:gemini-3.5-flash-lite")
    fallback = FakeModel("openai:gpt-5.4-nano")
    middleware = ObservableModelFallbackMiddleware(fallback)
    calls = []

    def handler(request):
        calls.append(request.model.model_name)
        if request.model is primary:
            raise TimeoutError("timed out")
        return "fallback response"

    assert middleware.wrap_model_call(_request(primary), handler) == "fallback response"
    calls.clear()
    assert middleware.wrap_model_call(_request(primary), handler) == "fallback response"
    assert calls == ["google_genai:gemini-3.5-flash-lite", "openai:gpt-5.4-nano"]
    ObservableModelFallbackMiddleware.reset_primary_breaker()
