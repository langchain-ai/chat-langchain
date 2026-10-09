"""Tests for primary-provider auth visibility during model fallback."""

import asyncio
import logging

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai.chat_models import GoogleInvalidRequestError
from langsmith import RunTree

from src.agent import config
from src.middleware.retry_middleware import ModelRetryMiddleware


class FakeModel(FakeListChatModel):
    model: str


@pytest.fixture(autouse=True)
def reset_log_limit(monkeypatch):
    monkeypatch.setattr(config, "_auth_failure_logged", False)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("with_run_tree", [False, True])
@pytest.mark.parametrize("failure", ["auth", "transient", "none"])
def test_fallback_visibility(monkeypatch, caplog, use_async, with_run_tree, failure):
    primary = FakeModel(model="gemini-3.5-flash-lite", responses=["primary"])
    fallback = FakeModel(model="gpt-5.4-nano", responses=["fallback"])
    root = RunTree(name="root")
    child = root.create_child(name="model", run_type="llm")
    child.parent_run = root
    monkeypatch.setattr(
        config.run_helpers,
        "get_current_run_tree",
        lambda: child if with_run_tree else None,
    )
    request = ModelRequest(model=primary, messages=[HumanMessage(content="Hi")])
    calls = []
    expected = ModelResponse(result=[AIMessage(content="success")])

    def handler(model_request):
        calls.append(model_request.model)
        if model_request.model is primary:
            if failure == "auth":
                raise GoogleInvalidRequestError(
                    "400 INVALID_ARGUMENT: API key not valid"
                )
            if failure == "transient":
                raise TimeoutError("timed out")
        return expected

    async def async_handler(model_request):
        return handler(model_request)

    caplog.set_level(logging.ERROR, logger=config.__name__)
    for _attempt in range(2):
        middleware = config._AuthAwareModelFallbackMiddleware(fallback)
        result = (
            asyncio.run(middleware.awrap_model_call(request, async_handler))
            if use_async
            else middleware.wrap_model_call(request, handler)
        )
        assert result is expected
    assert calls == ([primary] * 2 if failure == "none" else [primary, fallback] * 2)
    logs = [record for record in caplog.records if record.name == config.__name__]
    assert len(logs) == (1 if failure == "auth" else 0)
    if failure == "auth":
        assert "provider=google" in logs[0].message
        assert "model=gemini-3.5-flash-lite" in logs[0].message
    assert root.metadata.get("primary_model_auth_failed", False) is (
        failure == "auth" and with_run_tree
    )
    assert "primary_model_auth_failed" not in child.metadata


def test_invalid_key_skips_retries_and_serves_visible_fallback(monkeypatch, caplog):
    primary = FakeModel(model="gemini-3.5-flash-lite", responses=["primary"])
    fallback = FakeModel(model="gpt-5.4-nano", responses=["fallback"])
    middleware = config._AuthAwareModelFallbackMiddleware(fallback)
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    root = RunTree(name="root")
    monkeypatch.setattr(config.run_helpers, "get_current_run_tree", lambda: root)
    calls = []

    async def provider(model_request):
        calls.append(model_request.model)
        if model_request.model is primary:
            raise GoogleInvalidRequestError("API_KEY_INVALID")
        return ModelResponse(result=[AIMessage(content="fallback response")])

    async def fallback_handler(model_request):
        return await middleware.awrap_model_call(model_request, provider)

    caplog.set_level(logging.ERROR, logger=config.__name__)
    response = asyncio.run(
        retry.awrap_model_call(
            ModelRequest(model=primary, messages=[HumanMessage(content="Hi")]),
            fallback_handler,
        )
    )
    assert response.result[0].content == "fallback response"
    assert calls == [primary, fallback]
    assert root.metadata["primary_model_auth_failed"] is True
    assert "Primary model authentication failed" in caplog.text


def test_each_root_is_tagged_after_log_limit_is_reached(monkeypatch, caplog):
    primary = FakeModel(model="gemini-3.5-flash-lite", responses=["primary"])
    fallback = FakeModel(model="gpt-5.4-nano", responses=["fallback"])
    request = ModelRequest(model=primary, messages=[HumanMessage(content="Hi")])
    caplog.set_level(logging.ERROR, logger=config.__name__)

    def handler(model_request):
        if model_request.model is primary:
            raise GoogleInvalidRequestError("API_KEY_INVALID")
        return ModelResponse(result=[AIMessage(content="success")])

    for _attempt in range(2):
        root = RunTree(name="root")
        monkeypatch.setattr(config.run_helpers, "get_current_run_tree", lambda: root)
        config._AuthAwareModelFallbackMiddleware(fallback).wrap_model_call(
            request, handler
        )
        assert root.metadata["primary_model_auth_failed"] is True
    assert (
        len([record for record in caplog.records if record.name == config.__name__])
        == 1
    )


def test_secondary_auth_error_does_not_mark_primary(monkeypatch, caplog):
    primary = FakeModel(model="gemini-3.5-flash-lite", responses=["primary"])
    fallback = FakeModel(model="gpt-5.4-nano", responses=["fallback"])
    final = FakeModel(model="claude-haiku-4-5-20251001", responses=["final"])
    root = RunTree(name="root")
    monkeypatch.setattr(config.run_helpers, "get_current_run_tree", lambda: root)

    def handler(model_request):
        if model_request.model is primary:
            raise TimeoutError("timed out")
        if model_request.model is fallback:
            raise GoogleInvalidRequestError("API_KEY_INVALID")
        return ModelResponse(result=[AIMessage(content="success")])

    response = config._AuthAwareModelFallbackMiddleware(
        fallback, final
    ).wrap_model_call(
        ModelRequest(model=primary, messages=[HumanMessage(content="Hi")]), handler
    )
    assert response.result[0].content == "success"
    assert "primary_model_auth_failed" not in root.metadata
    assert not caplog.records
