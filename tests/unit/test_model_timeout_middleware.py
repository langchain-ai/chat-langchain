"""Tests for per-attempt model timeouts and retry/fallback behavior."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


class StallingChatModel(FakeListChatModel):
    calls: int = 0
    cancellations: int = 0

    async def _agenerate(self, *args, **kwargs):
        self.calls += 1
        try:
            await asyncio.sleep(60)
        finally:
            self.cancellations += 1
        return await super()._agenerate(*args, **kwargs)


def test_stalled_handler_times_out_and_is_cancelled():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    request = ModelRequest(model=FakeListChatModel(responses=["unused"]), messages=[])
    cancelled = False

    async def handler(received_request):
        nonlocal cancelled
        assert received_request is request
        try:
            await asyncio.sleep(60)
        finally:
            cancelled = True

    with pytest.raises(TimeoutError):
        asyncio.run(middleware.awrap_model_call(request, handler))

    assert cancelled


def test_fast_handler_returns_result_unchanged():
    middleware = ModelCallTimeoutMiddleware(timeout_s=1)
    request = ModelRequest(model=FakeListChatModel(responses=["unused"]), messages=[])
    response = ModelResponse(result=[AIMessage(content="answer")])
    handler = AsyncMock(return_value=response)

    assert asyncio.run(middleware.awrap_model_call(request, handler)) is response
    handler.assert_awaited_once_with(request)


def test_sync_handler_passes_through():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    request = ModelRequest(model=FakeListChatModel(responses=["unused"]), messages=[])
    response = ModelResponse(result=[AIMessage(content="answer")])
    handler = Mock(return_value=response)

    assert middleware.wrap_model_call(request, handler) is response
    handler.assert_called_once_with(request)


def test_timeout_middleware_is_innermost_in_docs_agent():
    from agent import docs_agent_middleware
    from src.agent import config

    assert docs_agent_middleware[-2] is config.model_fallback_middleware
    assert docs_agent_middleware[-1] is config.model_timeout_middleware
    assert config.model_timeout_middleware.timeout_s == config.MODEL_CALL_TIMEOUT_S
    assert config.default_model.timeout == config.MODEL_CALL_TIMEOUT_S


def test_retrying_models_receive_configured_timeout(monkeypatch):
    from src.agent import config

    init_model = Mock(return_value=FakeListChatModel(responses=["answer"]))
    monkeypatch.setattr(config, "init_chat_model", init_model)
    monkeypatch.setattr(config, "MODEL_CALL_TIMEOUT_S", 0.25)

    config._init_retrying_model(config.DEFAULT_MODEL.id)

    init_model.assert_called_once_with(model=config.DEFAULT_MODEL.id, timeout=0.25)


def test_timed_out_primary_invokes_fallback():
    primary = StallingChatModel(responses=["unused"])
    fallback = FakeListChatModel(responses=["fallback answer"])
    agent = create_agent(
        model=primary,
        middleware=[
            ModelRetryMiddleware(max_retries=1, initial_delay=0),
            ModelFallbackMiddleware(fallback),
            ModelCallTimeoutMiddleware(timeout_s=0.01),
        ],
    )

    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]})
    )

    assert result["messages"][-1].content == "fallback answer"
    assert primary.calls == primary.cancellations == 1


def test_timeout_applies_to_fallback_and_propagates_after_retries():
    primary = StallingChatModel(responses=["unused"])
    fallback = StallingChatModel(responses=["unused"])
    agent = create_agent(
        model=primary,
        middleware=[
            ModelRetryMiddleware(max_retries=1, initial_delay=0),
            ModelFallbackMiddleware(fallback),
            ModelCallTimeoutMiddleware(timeout_s=0.01),
        ],
    )

    with pytest.raises(TimeoutError):
        asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]}))

    assert primary.calls == primary.cancellations == 2
    assert fallback.calls == fallback.cancellations == 2
