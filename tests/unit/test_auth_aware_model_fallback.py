"""Tests for authentication-aware model fallback behavior."""

import os
from types import SimpleNamespace

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from langchain.agents.middleware import ModelRequest

from src.agent import config


class FakeRunTree:
    def __init__(self):
        self.metadata = {}
        self.tags = []

    def add_metadata(self, metadata):
        self.metadata.update(metadata)

    def add_tags(self, tags):
        self.tags.extend(tags)


def _middleware(monkeypatch):
    primary = SimpleNamespace(_llm_type="google_genai")
    fallback = SimpleNamespace(_llm_type="openai")
    middleware = config.AuthAwareModelFallbackMiddleware.__new__(
        config.AuthAwareModelFallbackMiddleware
    )
    middleware.models = [fallback]
    middleware.model_ids = ["openai:gpt-5.4-nano"]
    monkeypatch.setattr(config, "_auth_breaker_until", 0.0)
    return middleware, primary, fallback


def test_auth_failure_records_metadata_and_opens_breaker(monkeypatch):
    middleware, primary, fallback = _middleware(monkeypatch)
    run_tree = FakeRunTree()
    monkeypatch.setattr(config.ls, "get_current_run_tree", lambda: run_tree)
    request = ModelRequest(model=primary, messages=[])
    calls = []

    class AuthError(Exception):
        reason = "API_KEY_INVALID"

    def handler(current_request):
        calls.append(current_request.model)
        if current_request.model is primary:
            raise AuthError("invalid key")
        return "fallback response"

    assert middleware.wrap_model_call(request, handler) == "fallback response"
    assert calls == [primary, fallback]
    assert run_tree.metadata["model_fallback_reason"] == "auth"
    assert run_tree.metadata["model_fallback_serving_model"] == "openai:gpt-5.4-nano"
    assert "model-fallback-auth" in run_tree.tags

    calls.clear()
    assert middleware.wrap_model_call(request, handler) == "fallback response"
    assert calls == [fallback]


def test_transient_failure_falls_back_without_opening_breaker(monkeypatch):
    middleware, primary, fallback = _middleware(monkeypatch)
    request = ModelRequest(model=primary, messages=[])
    calls = []

    def handler(current_request):
        calls.append(current_request.model)
        if current_request.model is primary:
            raise TimeoutError("temporary outage")
        return "fallback response"

    assert middleware.wrap_model_call(request, handler) == "fallback response"
    assert calls == [primary, fallback]

    calls.clear()
    assert middleware.wrap_model_call(request, handler) == "fallback response"
    assert calls == [primary, fallback]
