"""Tests for model call retry middleware."""

import asyncio

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


@pytest.mark.anyio
async def test_model_call_timeout_is_retried_and_raised():
    """A timed out model call uses the configured retry budget."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return request

    middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0, timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        await middleware.awrap_model_call(object(), handler)

    assert calls == 2
