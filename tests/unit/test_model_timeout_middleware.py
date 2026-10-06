"""Tests for model timeout middleware."""

import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest

from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_timeout_middleware_raises_timeout_error():
    """A model call exceeding its deadline raises TimeoutError."""
    middleware = ModelTimeoutMiddleware(timeout=0.01)

    async def handler(_request):
        await asyncio.sleep(0.05)

    with pytest.raises(TimeoutError, match="model call exceeded 0.01s"):
        asyncio.run(middleware.awrap_model_call(object(), handler))


def test_model_timeout_retries_then_falls_back():
    """A timed out primary retries before fallback succeeds."""
    timeout_middleware = ModelTimeoutMiddleware(timeout=0.01)
    retry_middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    fallback_middleware = ModelFallbackMiddleware.__new__(ModelFallbackMiddleware)
    fallback_middleware.models = ["fallback"]
    primary_calls = 0
    fallback_calls = 0

    async def model_handler(request):
        nonlocal primary_calls, fallback_calls
        if request.model == "fallback":
            fallback_calls += 1
            return "fallback result"
        primary_calls += 1
        await asyncio.sleep(0.05)

    async def timed_handler(request):
        return await timeout_middleware.awrap_model_call(request, model_handler)

    async def retry_handler(request):
        return await retry_middleware.awrap_model_call(request, timed_handler)

    request = ModelRequest(model=object(), messages=[])
    result = asyncio.run(fallback_middleware.awrap_model_call(request, retry_handler))

    assert result == "fallback result"
    assert primary_calls == 2
    assert fallback_calls == 1
