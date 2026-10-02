"""Tests for startup model credential validation."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ["MODEL_STARTUP_AUTH_CHECK"] = "false"

from src.agent import config


def test_validate_configured_model_credentials_disables_tracing(monkeypatch):
    calls = []
    tracing_options = []

    class FakeModel:
        def invoke(self, messages, **kwargs):
            calls.append((messages, kwargs))

    monkeypatch.setattr(config, "init_chat_model", lambda **kwargs: FakeModel())
    monkeypatch.setenv("MODEL_STARTUP_AUTH_CHECK", "true")
    monkeypatch.setattr(
        config,
        "tracing_context",
        lambda **kwargs: tracing_options.append(kwargs) or _context_manager(),
    )
    monkeypatch.setattr(config, "API_KEYS", ["OPENAI_API_KEY"])
    monkeypatch.setattr(
        config,
        "MODELS",
        {
            "gpt-5.4-nano": SimpleNamespace(
                api_key_env="OPENAI_API_KEY", id="openai:gpt-5.4-nano"
            )
        },
    )
    monkeypatch.setattr(config, "DEFAULT_MODEL", config.MODELS["gpt-5.4-nano"])
    monkeypatch.setattr(config, "FALLBACK_MODELS", [])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config.validate_configured_model_credentials()

    assert len(calls) == 1
    assert calls[0][1] == {"max_tokens": 16}
    assert tracing_options == [{"enabled": False}]


def _context_manager():
    class ContextManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    return ContextManager()
