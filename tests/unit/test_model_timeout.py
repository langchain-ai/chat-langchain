"""Tests for bounded model construction and timeout retries."""

import asyncio

import httpx
import pytest

from src.agent import config
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_timed_model_maps_provider_timeout_arguments(monkeypatch):
    """Provider integrations receive their supported timeout argument."""
    calls = []

    def fake_init_chat_model(model, **kwargs):
        calls.append((model, kwargs))
        return object()

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)

    config.init_timed_chat_model("google_genai:gemini-test", temperature=0)
    config.init_timed_chat_model("openai:gpt-test")
    config.init_timed_chat_model("anthropic:claude-test")

    assert calls == [
        (
            "google_genai:gemini-test",
            {"temperature": 0, "request_timeout": config.MODEL_TIMEOUT_SECONDS},
        ),
        (
            "openai:gpt-test",
            {
                "timeout": config.MODEL_TIMEOUT_SECONDS,
                "stream_chunk_timeout": config.MODEL_TIMEOUT_SECONDS,
            },
        ),
        (
            "anthropic:claude-test",
            {"timeout": config.MODEL_TIMEOUT_SECONDS},
        ),
    ]


def test_graph_uses_timed_primary_and_fallback_models():
    """Primary and middleware fallback models carry the configured timeout."""

    def configured_timeout(model):
        return next(
            value
            for name in ("timeout", "request_timeout", "default_request_timeout")
            if (value := getattr(model, name, None)) is not None
        )

    assert configured_timeout(config.default_model) == config.MODEL_TIMEOUT_SECONDS
    assert configured_timeout(config.guardrails_model) == config.MODEL_TIMEOUT_SECONDS
    assert all(
        configured_timeout(model) == config.MODEL_TIMEOUT_SECONDS
        for model in config.model_fallback_middleware.models
    )


def test_model_retry_retries_timeout_then_raises():
    """Timeout failures are retried and the final exception is preserved."""
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("model request timed out")

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(middleware.awrap_model_call(object(), handler))

    assert calls == 3
