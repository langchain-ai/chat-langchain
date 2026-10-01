from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest

from src.agent import config


def _request() -> ModelRequest:
    return ModelRequest(model=SimpleNamespace(model=config.DEFAULT_MODEL.id), messages=[])


def _middleware(*models: str) -> config.CredentialAwareModelFallbackMiddleware:
    middleware = config.CredentialAwareModelFallbackMiddleware(*models)
    middleware.models = [SimpleNamespace(model=model) for model in models]
    return middleware


@pytest.mark.parametrize(
    "exception, expected",
    [
        (SimpleNamespace(status_code=401), True),
        (SimpleNamespace(code=403), True),
        (SimpleNamespace(response=SimpleNamespace(status_code=401)), True),
        (RuntimeError("API_KEY_INVALID"), True),
        (RuntimeError("PERMISSION_DENIED"), True),
        (RuntimeError("API key not valid"), True),
        (RuntimeError("invalid x-api-key"), True),
        (RuntimeError("temporary upstream failure"), False),
    ],
)
def test_credential_error_classification(exception, expected):
    assert config._is_credential_error(exception) is expected


def test_credential_failure_stamps_run_tree_and_served_model(monkeypatch, caplog):
    run_tree = SimpleNamespace(metadata={}, add_metadata=lambda values: run_tree.metadata.update(values))
    monkeypatch.setattr(config, "get_current_run_tree", lambda: run_tree)
    monkeypatch.setattr(config, "_credential_error_logged", False)
    middleware = _middleware("openai:gpt-5.4-nano")

    def handler(request):
        if request.model.model == config.DEFAULT_MODEL.id:
            raise RuntimeError("API_KEY_INVALID")
        return "fallback response"

    with caplog.at_level("ERROR", logger=config.logger.name):
        assert middleware.wrap_model_call(_request(), handler) == "fallback response"

    assert run_tree.metadata == {
        "primary_model_failed": "auth",
        "primary_model": config.DEFAULT_MODEL.id,
        "served_model": "openai:gpt-5.4-nano",
    }
    assert "GOOGLE_API_KEY" in caplog.text


def test_credential_failure_logs_once_per_process(monkeypatch, caplog):
    monkeypatch.setattr(config, "get_current_run_tree", lambda: None)
    monkeypatch.setattr(config, "_credential_error_logged", False)
    middleware = _middleware("openai:gpt-5.4-nano")

    def handler(request):
        if request.model.model == config.DEFAULT_MODEL.id:
            raise RuntimeError("PERMISSION_DENIED")
        return "fallback response"

    with caplog.at_level("ERROR", logger=config.logger.name):
        middleware.wrap_model_call(_request(), handler)
        middleware.wrap_model_call(_request(), handler)

    assert caplog.text.count("failed credential validation") == 1


def test_non_credential_errors_keep_fallback_sequence(monkeypatch):
    monkeypatch.setattr(config, "get_current_run_tree", lambda: None)
    middleware = _middleware("openai:gpt-5.4-nano", "anthropic:claude-haiku-4-5-20251001")
    attempts = []

    def handler(request):
        attempts.append(request.model.model)
        if len(attempts) < 3:
            raise RuntimeError("temporary upstream failure")
        return "fallback response"

    assert middleware.wrap_model_call(_request(), handler) == "fallback response"
    assert attempts == [
        config.DEFAULT_MODEL.id,
        "openai:gpt-5.4-nano",
        "anthropic:claude-haiku-4-5-20251001",
    ]
