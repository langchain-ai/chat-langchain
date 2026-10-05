"""Tests for model call timeout middleware."""

import asyncio
import os
import time

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware

for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.setdefault(key, "test-key")


class FakeModel(BaseChatModel):
    """Minimal chat model used to identify fallback attempts."""

    @property
    def _llm_type(self):
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return ChatResult(generations=[])


def test_timeout_middleware_raises_after_deadline():
    """A stalled model call raises TimeoutError at its deadline."""
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.02)
    request = ModelRequest(model="primary", messages=[])

    async def handler(_request):
        await asyncio.sleep(1)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="primary"):
        asyncio.run(middleware.awrap_model_call(request, handler))

    assert time.monotonic() - started < 0.5


def test_timeout_allows_fallback_model_to_run():
    """A timed-out primary lets the outer fallback middleware continue."""
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.02)
    primary = FakeModel()
    fallback = FakeModel()
    fallback_middleware = ModelFallbackMiddleware(fallback)
    request = ModelRequest(model=primary, messages=[])
    calls = []

    async def model_handler(model_request):
        calls.append(model_request.model)
        if model_request.model is primary:
            await asyncio.sleep(1)
        return AIMessage(content="fallback response")

    async def timeout_handler(model_request):
        return await timeout_middleware.awrap_model_call(model_request, model_handler)

    result = asyncio.run(fallback_middleware.awrap_model_call(request, timeout_handler))

    assert result.content == "fallback response"
    assert calls == [primary, fallback]


def test_timeout_middleware_is_innermost_docs_agent_middleware():
    """The timeout wraps each provider attempt beneath model fallback."""
    from agent import docs_agent_middleware
    from src.agent import config

    assert docs_agent_middleware[-1] is config.model_timeout_middleware
    assert docs_agent_middleware[-2] is config.model_fallback_middleware
