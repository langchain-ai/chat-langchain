"""Tests for model call retry timeouts."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_call_timeout_retries_then_raises():
    """A stalled model call should retry and then raise its timeout."""
    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout_seconds=0.01,
    )
    attempts = 0

    async def handler(request):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    started_at = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(None, handler))
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert attempts == 3


def test_model_call_timeout_retries_until_success():
    """A later successful attempt should be returned after a timeout."""
    middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        timeout_seconds=0.01,
    )
    attempts = 0
    response = SimpleNamespace(response_metadata={})

    async def handler(request):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.Event().wait()
        return response

    result = asyncio.run(middleware.awrap_model_call(None, handler))

    assert result is response
    assert attempts == 2
