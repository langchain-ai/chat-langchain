"""Tests for model credential preflight behavior."""

import importlib

import pytest


class AuthFailure(Exception):
    pass


def _load_config(monkeypatch, *, strict: str | None = None):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("MODEL_PREFLIGHT", raising=False)
    if strict is None:
        monkeypatch.delenv("STRICT_MODEL_PREFLIGHT", raising=False)
    else:
        monkeypatch.setenv("STRICT_MODEL_PREFLIGHT", strict)
    import src.agent.config as config

    return importlib.reload(config)


def test_preflight_auth_failure_logs_without_raising(monkeypatch, caplog):
    config = _load_config(monkeypatch)

    class FakeModel:
        def invoke(self, *_args, **_kwargs):
            raise AuthFailure("API_KEY_INVALID")

    config.default_model = FakeModel()
    monkeypatch.setenv("MODEL_PREFLIGHT", "1")

    with caplog.at_level("ERROR"):
        config._run_model_preflight()

    assert "authentication preflight failed" in caplog.text


def test_preflight_auth_failure_raises_in_strict_mode(monkeypatch):
    config = _load_config(monkeypatch, strict="1")

    class FakeModel:
        def invoke(self, *_args, **_kwargs):
            raise AuthFailure("HTTP 401")

    config.default_model = FakeModel()
    monkeypatch.setenv("MODEL_PREFLIGHT", "1")

    with pytest.raises(AuthFailure):
        config._run_model_preflight()
