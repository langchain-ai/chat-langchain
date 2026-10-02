import importlib
import sys
from contextlib import contextmanager


def _load_config(monkeypatch):
    monkeypatch.setenv("MODEL_STARTUP_AUTH_CHECK", "false")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    sys.modules.pop("src.agent.config", None)
    return importlib.import_module("src.agent.config")


class _Models:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def retrieve(self, model):
        self.calls.append(model)
        if self.error:
            raise self.error


class _OpenAIClient:
    def __init__(self, models, **kwargs):
        self.models = models


class _ProviderError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def _configure_openai(monkeypatch, config, error=None):
    models = _Models(error)
    monkeypatch.setenv("MODEL_STARTUP_AUTH_CHECK", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        config, "OpenAI", lambda **kwargs: _OpenAIClient(models, **kwargs)
    )
    config._credential_probe_completed = False
    config._credential_probe_results.clear()
    return models


def test_output_limit_probe_error_is_inconclusive_and_untraced(monkeypatch):
    config = _load_config(monkeypatch)
    models = _configure_openai(
        monkeypatch,
        config,
        _ProviderError(
            "Could not finish the message because max_tokens or model output limit was reached",
            400,
        ),
    )
    tracing_states = []

    @contextmanager
    def tracing_disabled(**kwargs):
        tracing_states.append(kwargs)
        yield

    monkeypatch.setattr(config, "tracing_context", tracing_disabled)

    assert config.validate_provider_authentication() == {"openai": None}
    assert models.calls == ["gpt-5.4-nano"]
    assert tracing_states == [{"enabled": False}]


def test_unauthorized_probe_error_marks_credential_invalid(monkeypatch):
    config = _load_config(monkeypatch)
    _configure_openai(monkeypatch, config, _ProviderError("unauthorized", 401))

    assert config.validate_provider_authentication() == {"openai": False}


def test_probe_runs_at_most_once_per_process(monkeypatch):
    config = _load_config(monkeypatch)
    models = _configure_openai(monkeypatch, config)

    assert config.validate_provider_authentication() == {"openai": True}
    assert config.validate_provider_authentication() == {"openai": True}
    assert models.calls == ["gpt-5.4-nano"]
