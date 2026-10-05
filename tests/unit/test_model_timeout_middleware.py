"""Tests for model call timeout middleware."""

import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def test_model_call_timeout_raises_timeout_error():
    """A sleeping model handler raises after the configured timeout."""
    middleware = ModelCallTimeoutMiddleware(timeout_seconds=0.01)
    model = FakeListChatModel(responses=["unused"])
    request = ModelRequest(model=model, messages=[HumanMessage(content="hello")])

    async def handler(_request):
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError):
        asyncio.run(middleware.awrap_model_call(request, handler))


def test_model_fallback_retries_after_timeout():
    """A timed-out primary model falls back to the next model."""
    primary = FakeListChatModel(responses=["unused"])
    fallback = FakeListChatModel(responses=["unused"])
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_seconds=0.01)
    fallback_middleware = ModelFallbackMiddleware(primary, fallback)
    request = ModelRequest(
        model=primary,
        messages=[HumanMessage(content="hello")],
    )

    async def terminal_handler(current_request):
        if current_request.model is primary:
            await asyncio.sleep(1)
        return AIMessage(content="fallback response")

    async def timeout_handler(current_request):
        return await timeout_middleware.awrap_model_call(
            current_request, terminal_handler
        )

    result = asyncio.run(fallback_middleware.awrap_model_call(request, timeout_handler))

    assert result.content == "fallback response"
