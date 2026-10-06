import asyncio
import time

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_sync_model_call_times_out():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

    def handler(request):  # noqa: ARG001
        time.sleep(0.1)
        return "answer"

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline"):
        middleware.wrap_model_call(None, handler)

    assert time.monotonic() - started < 0.08


def test_async_model_call_times_out():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

    async def handler(request):  # noqa: ARG001
        await asyncio.sleep(0.1)
        return "answer"

    with pytest.raises(TimeoutError, match="deadline"):
        asyncio.run(middleware.awrap_model_call(None, handler))


def test_async_timeout_retries_and_falls_back_to_answer():
    timeout = ModelCallTimeoutMiddleware(timeout_s=0.01)
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0, backoff_factor=1)
    primary_model = object()
    fallback_model = object()
    fallback = ModelFallbackMiddleware(fallback_model)
    calls = []
    fallback_attempts = 0

    async def model_call(request):
        if request.model is primary_model:
            calls.append("primary")
            await asyncio.sleep(0.1)
            return "unreachable"
        calls.append("fallback")
        nonlocal fallback_attempts
        fallback_attempts += 1
        if fallback_attempts == 1:
            await asyncio.sleep(0.1)
        return "fallback answer"

    async def model_handler(request):
        return await timeout.awrap_model_call(request, model_call)

    async def fallback_handler(request):
        return await fallback.awrap_model_call(request, model_handler)

    request = ModelRequest(model=primary_model, messages=[])

    assert (
        asyncio.run(retry.awrap_model_call(request, fallback_handler))
        == "fallback answer"
    )
    assert calls == ["primary", "fallback", "primary", "fallback"]


def test_sync_timeout_retries_and_falls_back_to_answer():
    timeout = ModelCallTimeoutMiddleware(timeout_s=0.01)
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0, backoff_factor=1)
    primary_model = object()
    fallback_model = object()
    fallback = ModelFallbackMiddleware(fallback_model)
    calls = []
    fallback_attempts = 0

    def model_call(request):
        if request.model is primary_model:
            calls.append("primary")
            time.sleep(0.1)
            return "unreachable"
        calls.append("fallback")
        nonlocal fallback_attempts
        fallback_attempts += 1
        if fallback_attempts == 1:
            time.sleep(0.1)
        return "fallback answer"

    def model_handler(request):
        return timeout.wrap_model_call(request, model_call)

    def fallback_handler(request):
        return fallback.wrap_model_call(request, model_handler)

    request = ModelRequest(model=primary_model, messages=[])

    assert retry.wrap_model_call(request, fallback_handler) == "fallback answer"
    assert calls == ["primary", "fallback", "primary", "fallback"]
