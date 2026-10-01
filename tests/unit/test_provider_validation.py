import logging
import os

import pytest

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from src.agent import config


class FakeModel:
    def __init__(self, error=None):
        self.error = error

    def invoke(self, _message):
        if self.error:
            raise self.error
        return object()


def test_primary_authentication_failure_raises_without_logging_secret(
    monkeypatch, caplog
):
    secret = "super-secret-google-key"
    monkeypatch.delenv("SKIP_PROVIDER_KEY_CHECK", raising=False)
    monkeypatch.setattr(
        config,
        "init_chat_model",
        lambda model: FakeModel(
            ValueError(f"API_KEY_INVALID: {secret}")
            if model == config.DEFAULT_MODEL.id
            else None
        ),
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(config.ProviderKeyValidationError),
    ):
        config.validate_provider_keys()

    assert "GOOGLE_API_KEY" in caplog.text
    assert secret not in caplog.text


def test_skip_provider_key_check_bypasses_provider_calls(monkeypatch):
    monkeypatch.setenv("SKIP_PROVIDER_KEY_CHECK", "1")
    called = False

    def init_model(_model):
        nonlocal called
        called = True
        return FakeModel()

    monkeypatch.setattr(config, "init_chat_model", init_model)

    config.validate_provider_keys()

    assert not called
