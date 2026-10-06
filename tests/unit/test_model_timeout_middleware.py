"""Tests for per-call model timeout behavior."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_sleeping_model_times_out():
    """A model call exceeding the deadline raises TimeoutError."""
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

    async def sleeping_handler(request):  # noqa: ARG001
        await asyncio.sleep(1)
        return ModelResponse(result=[])

    with pytest.raises(TimeoutError, match=r"model call exceeded 0.01s"):
        asyncio.run(middleware.awrap_model_call(object(), sleeping_handler))


def test_hanging_primary_returns_fallback_within_deadline():
    """A timed-out primary is retried and then replaced by the fallback."""
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    retry_middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0.001,
    )
    primary_model = MagicMock(spec=BaseChatModel)
    fallback_model = MagicMock(spec=BaseChatModel)
    fallback_middleware = ModelFallbackMiddleware(fallback_model)
    request = ModelRequest(model=primary_model, messages=[])
    calls = []

    async def handler(current_request):
        calls.append(current_request.model)
        if current_request.model is fallback_model:
            return ModelResponse(result=[])
        await asyncio.sleep(1)
        return ModelResponse(result=[])

    async def invoke():
        return await fallback_middleware.awrap_model_call(
            request,
            lambda current_request: retry_middleware.awrap_model_call(
                current_request,
                lambda nested_request: timeout_middleware.awrap_model_call(
                    nested_request,
                    handler,
                ),
            ),
        )

    started_at = time.monotonic()
    result = asyncio.run(invoke())
    elapsed = time.monotonic() - started_at

    assert isinstance(result, ModelResponse)
    assert calls == [request.model, request.model, fallback_model]
    assert elapsed < 0.5
