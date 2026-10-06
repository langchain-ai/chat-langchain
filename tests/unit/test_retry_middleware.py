"""Tests for model call timeout handling."""

import asyncio
import time

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_retry_middleware_times_out_each_attempt():
    """A stalled model call is retried and then raises its timeout."""
    calls = 0

    async def slow_handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        timeout_seconds=0.01,
    )

    started_at = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(None, slow_handler))

    assert calls == 2
    assert time.monotonic() - started_at < 0.5


def test_model_retry_middleware_returns_fast_handler_result():
    """A model call that finishes before the deadline succeeds."""

    async def fast_handler(request):  # noqa: ARG001
        return "result"

    middleware = ModelRetryMiddleware(timeout_seconds=0.1)

    result = asyncio.run(middleware.awrap_model_call(None, fast_handler))

    assert result == "result"
