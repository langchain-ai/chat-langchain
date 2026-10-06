"""Tests for model call timeout and retry behavior."""

import asyncio

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_timeout_is_raised_after_retry_exhaustion():
    """Timeouts are retried and re-raised after the retry budget is exhausted."""
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout_seconds=0.01,
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(None, handler))

    assert calls == 3


def test_timeout_can_recover_on_retry():
    """A later successful attempt returns after an earlier timeout."""
    calls = 0
    response = object()

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)
        return response

    middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        timeout_seconds=0.01,
    )

    assert asyncio.run(middleware.awrap_model_call(None, handler)) is response
    assert calls == 2
