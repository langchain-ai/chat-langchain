import os

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agent import config
from src.middleware import model_fallback_middleware as fallback_module
from src.middleware.model_fallback_middleware import (
    CredentialAwareModelFallbackMiddleware,
    is_permanent_credential_error,
)


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _fake_model(model_id: str) -> GenericFakeChatModel:
    model = GenericFakeChatModel(messages=iter(["fallback response"]))
    object.__setattr__(model, "model", model_id)
    return model


@pytest.mark.parametrize(
    ("provider", "message", "status_code", "expected"),
    [
        ("google_genai", "API_KEY_INVALID", 400, True),
        ("google_genai", "PERMISSION_DENIED", 403, True),
        ("openai", "invalid_api_key", 401, True),
        ("anthropic", "authentication_error", 401, True),
        ("google_genai", "rate limited", 429, False),
        ("openai", "server error", 500, False),
    ],
)
def test_credential_error_classification(
    provider: str, message: str, status_code: int, expected: bool
):
    assert is_permanent_credential_error(
        ProviderError(message, status_code), provider
    ) is expected


def test_primary_credential_failure_is_suppressed_until_expiry(monkeypatch):
    now = [100.0]
    primary = _fake_model("google_genai:gemini-3.5-flash-lite")
    fallback = _fake_model("openai:gpt-5.4-nano")
    middleware = CredentialAwareModelFallbackMiddleware(
        fallback, cooldown_seconds=10, clock=lambda: now[0]
    )
    metadata = {}
    monkeypatch.setattr(fallback_module, "get_config", lambda: {"metadata": metadata})
    calls = {"primary": 0, "fallback": 0}
    metadata_seen_by_fallback = []

    def handler(request):
        if request.model is primary:
            calls["primary"] += 1
            raise ProviderError("API_KEY_INVALID", 400)
        calls["fallback"] += 1
        metadata_seen_by_fallback.append(dict(metadata))
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(model=primary, messages=[])
    middleware.wrap_model_call(request, handler)
    middleware.wrap_model_call(request, handler)
    assert calls == {"primary": 1, "fallback": 2}
    assert metadata == {
        "fallback_served": "openai:gpt-5.4-nano",
        "primary_model_failed": True,
    }
    assert metadata_seen_by_fallback == [metadata, metadata]

    now[0] = 111.0
    middleware.wrap_model_call(request, handler)
    assert calls == {"primary": 2, "fallback": 3}


def test_transient_primary_failure_does_not_enter_cooldown():
    primary = _fake_model("google_genai:gemini-3.5-flash-lite")
    fallback = _fake_model("openai:gpt-5.4-nano")
    middleware = CredentialAwareModelFallbackMiddleware(
        fallback, cooldown_seconds=10
    )
    calls = {"primary": 0, "fallback": 0}

    def handler(request):
        if request.model is primary:
            calls["primary"] += 1
            raise ProviderError("rate limited", 429)
        calls["fallback"] += 1
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(model=primary, messages=[])
    middleware.wrap_model_call(request, handler)
    middleware.wrap_model_call(request, handler)
    assert calls == {"primary": 2, "fallback": 2}


def test_validation_skips_missing_keys_and_deduplicates_providers(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []

    class FakeModel:
        def invoke(self, message):
            calls.append(message)

    def fake_init_chat_model(*, model):
        calls.append(model)
        return FakeModel()

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    assert config.validate_provider_credentials() == {}
    assert calls == [
        "google_genai:gemini-3.5-flash-lite",
        "ping",
        "openai:gpt-5.4-nano",
        "ping",
    ]


def test_validation_reports_rejected_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    class FakeModel:
        def invoke(self, message):
            raise ProviderError("API_KEY_INVALID", 400)

    monkeypatch.setattr(config, "init_chat_model", lambda **kwargs: FakeModel())
    assert config.validate_provider_credentials() == {"google": "GOOGLE_API_KEY"}
