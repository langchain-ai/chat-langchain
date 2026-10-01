import logging
from types import SimpleNamespace

import pytest

from src.agent.config import ModelConfig, validate_primary_model_credential
from src.middleware.retry_middleware import (
    classify_provider_authentication_error,
    is_provider_authentication_error,
)


class ProviderError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def test_classifies_google_api_key_invalid_reason():
    exception = ProviderError(
        "INVALID_ARGUMENT",
        status_code=400,
        body={"error": {"details": [{"reason": "API_KEY_INVALID"}]}},
    )

    assert (
        classify_provider_authentication_error(exception, "google") == "API_KEY_INVALID"
    )


def test_classifies_openai_and_anthropic_401_errors():
    assert is_provider_authentication_error(
        ProviderError("authentication_error", status_code=401), "openai"
    )
    assert is_provider_authentication_error(
        ProviderError("authentication_error", status_code=401), "anthropic"
    )


def test_classifies_http_403_permission_error():
    assert (
        classify_provider_authentication_error(
            ProviderError("permission denied", status_code=403)
        )
        == "HTTP_403_PERMISSION"
    )


def test_does_not_classify_non_authentication_400_error():
    assert not is_provider_authentication_error(
        ProviderError("invalid request shape", status_code=400), "google"
    )


def test_missing_primary_credential_fails_in_strict_mode(monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    model_config = ModelConfig("test:model", "Test", "test", "TEST_API_KEY")

    with pytest.raises(RuntimeError, match="Missing primary model credential"):
        validate_primary_model_credential(model_config, strict=True)


def test_auth_invalid_primary_credential_fails_in_strict_mode(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "not-empty")
    model_config = ModelConfig("test:model", "Test", "google", "TEST_API_KEY")

    def model_factory(_model):
        return SimpleNamespace(
            invoke=lambda _prompt: (_ for _ in ()).throw(
                ProviderError(
                    "INVALID_ARGUMENT",
                    status_code=400,
                    body={"reason": "API_KEY_INVALID"},
                )
            )
        )

    with pytest.raises(RuntimeError, match="API_KEY_INVALID"):
        validate_primary_model_credential(model_config, model_factory, strict=True)


def test_non_strict_validation_logs_and_continues(monkeypatch, caplog):
    monkeypatch.setenv("TEST_API_KEY", "not-empty")
    model_config = ModelConfig("test:model", "Test", "google", "TEST_API_KEY")

    def model_factory(_model):
        return SimpleNamespace(
            invoke=lambda _prompt: (_ for _ in ()).throw(
                ProviderError("permission denied", status_code=403)
            )
        )

    with caplog.at_level(logging.ERROR):
        validate_primary_model_credential(model_config, model_factory, strict=False)

    assert "TEST_API_KEY" not in caplog.text
    assert "HTTP_403_PERMISSION" in caplog.text
