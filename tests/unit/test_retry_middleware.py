"""Tests for model call timeout and retry behavior."""

import asyncio
import time

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model: FakeChatModel) -> ModelRequest:
    return ModelRequest(model=model, messages=[])


def test_timeout_retries_then_falls_back_with_per_attempt_deadline():
    """A stalled primary is retried before fallback runs."""
    primary = FakeChatModel(name="primary")
    fallback = FakeChatModel(name="fallback")
    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout=0.01,
    )
    fallback_middleware = ModelFallbackMiddleware(fallback)
    calls = {"primary": 0, "fallback": 0}

    async def handler(request: ModelRequest) -> ModelResponse:
        if request.model is primary:
            calls["primary"] += 1
            await asyncio.Future()
        calls["fallback"] += 1
        return ModelResponse(result=[AIMessage(content="fallback response")])

    async def run() -> ModelResponse:
        async def retry_handler(request: ModelRequest) -> ModelResponse:
            return await retry.awrap_model_call(request, handler)

        return await fallback_middleware.awrap_model_call(
            _request(primary), retry_handler
        )

    started = time.perf_counter()
    response = asyncio.run(run())
    elapsed = time.perf_counter() - started

    assert response.result[0].content == "fallback response"
    assert calls == {"primary": 3, "fallback": 1}
    assert elapsed < 0.2


def test_fast_handler_is_called_once():
    """A fast model response is not delayed or retried."""
    model = FakeChatModel(name="fast")
    retry = ModelRetryMiddleware(max_retries=2, timeout=0.01)
    calls = 0

    async def handler(request: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="fast response")])

    response = asyncio.run(retry.awrap_model_call(_request(model), handler))

    assert response.result[0].content == "fast response"
    assert calls == 1
