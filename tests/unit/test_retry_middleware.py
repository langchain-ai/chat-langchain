"""Tests for model call retry timeouts."""

import asyncio
import time

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_call_timeout_retries_then_raises():
    """A never-ending model call is cancelled after its retry budget."""
    middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0.01,
        timeout_seconds=0.01,
    )

    async def never_returns(request):  # noqa: ARG001
        await asyncio.Event().wait()

    started_at = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(object(), never_returns))
    assert time.monotonic() - started_at < 0.2
