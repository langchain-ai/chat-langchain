"""Tests for model call timeout middleware."""

import asyncio
import importlib
import time

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_timeout_middleware_reads_environment_override(monkeypatch):
    monkeypatch.setenv("MODEL_CALL_TIMEOUT_SECONDS", "1.25")
    import src.middleware.model_timeout_middleware as timeout_module

    timeout_module = importlib.reload(timeout_module)

    assert timeout_module.ModelTimeoutMiddleware().timeout == 1.25
    monkeypatch.delenv("MODEL_CALL_TIMEOUT_SECONDS")
    importlib.reload(timeout_module)


def test_timeout_retries_then_falls_back_within_deadline():
    from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware

    primary = object()
    fallback = object()
    request = ModelRequest(model=primary, messages=[])
    timeout_middleware = ModelTimeoutMiddleware(timeout=0.01)
    retry_middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    attempts = 0

    async def exercise():
        nonlocal attempts

        async def handler(current_request):
            nonlocal attempts
            if current_request.model is primary:
                attempts += 1
                await asyncio.sleep(1)
            return ModelResponse(result=[AIMessage(content="fallback")])

        async def retry_handler(current_request):
            return await retry_middleware.awrap_model_call(
                current_request,
                lambda next_request: timeout_middleware.awrap_model_call(
                    next_request, handler
                ),
            )

        started = time.monotonic()
        response = await fallback_middleware.awrap_model_call(request, retry_handler)
        return response, started

    response, started = asyncio.run(exercise())

    assert response.result[0].content == "fallback"
    assert attempts == 2
    assert time.monotonic() - started < 0.3


def test_fast_model_succeeds_unchanged():
    from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware

    middleware = ModelTimeoutMiddleware(timeout=0.1)
    expected = ModelResponse(result=[AIMessage(content="ok")])

    async def exercise():
        async def handler(request):
            return expected

        return await middleware.awrap_model_call(
            ModelRequest(model=object(), messages=[]), handler
        )

    response = asyncio.run(exercise())

    assert response is expected
