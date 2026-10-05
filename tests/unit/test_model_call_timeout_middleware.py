"""Tests for bounded model calls and fallback behavior."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.middleware.retry_middleware import ModelCallTimeoutMiddleware


@pytest.fixture
def model_request():
    return SimpleNamespace(model=SimpleNamespace(model="test-model"))


def test_async_model_call_timeout_raises_within_bound(model_request):
    middleware = ModelCallTimeoutMiddleware(timeout=0.02)

    async def handler(_request):
        await asyncio.sleep(1)

    started = time.monotonic()

    async def run():
        with pytest.raises(TimeoutError):
            await middleware.awrap_model_call(model_request, handler)

    asyncio.run(run())

    assert time.monotonic() - started < 0.5


def test_sync_model_call_timeout_raises_within_bound(model_request):
    middleware = ModelCallTimeoutMiddleware(timeout=0.02)

    def handler(_request):
        time.sleep(1)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        middleware.wrap_model_call(model_request, handler)

    assert time.monotonic() - started < 0.5


def test_timeout_allows_fallback_model_to_respond():
    primary = SimpleNamespace(model="primary-model")
    fallback = SimpleNamespace(model="fallback-model")
    request = ModelRequest(model=primary, messages=[], tools=[])
    timeout_middleware = ModelCallTimeoutMiddleware(timeout=0.02)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    fallback_middleware.models = [fallback]

    async def handler(current_request):
        if current_request.model is primary:
            await asyncio.sleep(1)
            return ModelResponse(result=[])
        return ModelResponse(result=[])

    async def run():
        return await fallback_middleware.awrap_model_call(
            request,
            lambda current_request: timeout_middleware.awrap_model_call(
                current_request, handler
            ),
        )

    response = asyncio.run(run())

    assert response.result == []
