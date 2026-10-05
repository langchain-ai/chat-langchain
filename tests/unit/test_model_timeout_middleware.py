"""Tests for model call timeout middleware."""

import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def test_model_call_timeout_raises_for_hung_handler():
    async def handler(request):
        await asyncio.Event().wait()

    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    request = ModelRequest(model=FakeChatModel(), messages=[])

    with pytest.raises(TimeoutError):
        asyncio.run(middleware.awrap_model_call(request, handler))


def test_model_call_timeout_allows_fallback_after_hung_primary():
    primary = FakeChatModel(name="primary")
    fallback = FakeChatModel(name="fallback")
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    request = ModelRequest(model=primary, messages=[])
    calls = []

    async def handler(model_request):
        calls.append(model_request.model.name)
        if model_request.model is primary:
            await asyncio.Event().wait()
        return ModelResponse(result=[AIMessage(content="fallback response")])

    async def composed_handler(model_request):
        return await timeout_middleware.awrap_model_call(model_request, handler)

    response = asyncio.run(
        fallback_middleware.awrap_model_call(request, composed_handler)
    )

    assert response.result[0].content == "fallback response"
    assert calls == ["primary", "fallback"]
