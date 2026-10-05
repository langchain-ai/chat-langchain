"""Tests for model call timeout, retry, and fallback behavior."""

import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model):
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="Hello")],
    )


def test_timeout_retries_then_raises():
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        await asyncio.sleep(10)

    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout=0.01,
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(_request(FakeChatModel()), handler))

    assert calls == 3


def test_timeout_allows_fallback_model_to_respond():
    primary = FakeChatModel()
    fallback = FakeChatModel()
    retry = ModelRetryMiddleware(max_retries=0, timeout=0.01)
    fallback_middleware = ModelFallbackMiddleware(primary, fallback)

    async def handler(request):
        if request.model is primary:
            await asyncio.sleep(10)
        return ModelResponse([AIMessage(content="fallback response")])

    async def invoke(request):
        return await retry.awrap_model_call(request, handler)

    result = asyncio.run(
        fallback_middleware.awrap_model_call(_request(primary), invoke)
    )

    assert result.result[0].content == "fallback response"
