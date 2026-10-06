"""Tests for model timeout middleware."""

import asyncio
import time

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_timeout_raises_for_unresolved_call():
    """An unresolved model call should terminate at the configured deadline."""
    calls = 0
    timeout = ModelTimeoutMiddleware(0.01)

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(
            timeout.awrap_model_call(
                ModelRequest(model=object(), messages=[]),
                handler,
            )
        )

    assert calls == 1
    assert time.monotonic() - started < 0.5


def test_timeout_retries_then_uses_fallback():
    """Timeouts should exhaust primary retries before fallback runs."""
    primary = object()
    fallback = object()
    calls = {primary: 0, fallback: 0}
    timeout = ModelTimeoutMiddleware(0.01)
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    fallback_middleware = ModelFallbackMiddleware(fallback)

    async def terminal_handler(request):
        calls[request.model] += 1
        if request.model is primary:
            await asyncio.Event().wait()
        return ModelResponse(result=[])

    async def retry_handler(request):
        return await retry.awrap_model_call(
            request,
            lambda nested_request: timeout.awrap_model_call(
                nested_request,
                terminal_handler,
            ),
        )

    started = time.monotonic()
    result = asyncio.run(
        fallback_middleware.awrap_model_call(
            ModelRequest(model=primary, messages=[]),
            retry_handler,
        )
    )

    assert result.result == []
    assert calls == {primary: 2, fallback: 1}
    assert time.monotonic() - started < 0.5
