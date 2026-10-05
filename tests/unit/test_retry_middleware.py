"""Tests for model call timeout and retry behavior."""

import asyncio
import time

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


class FakeModel:
    """Minimal model identity for fallback middleware tests."""

    _llm_type = "fake"


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="hello")])


def test_hung_model_times_out_and_falls_back_within_bound():
    async def run_test():
        primary = FakeModel()
        fallback = FakeModel()
        fallback_middleware = ModelFallbackMiddleware(fallback)
        retry_middleware = ModelRetryMiddleware(
            max_retries=2,
            initial_delay=0,
            timeout_seconds=0.01,
        )
        calls = []

        async def handler(request):
            calls.append(request.model)
            if request.model is primary:
                await asyncio.Event().wait()
            return ModelResponse(result=[AIMessage(content="fallback")])

        async def retry_handler(request):
            return await retry_middleware.awrap_model_call(request, handler)

        started_at = time.monotonic()
        result = await fallback_middleware.awrap_model_call(
            _request(primary), retry_handler
        )

        assert result.result[0].content == "fallback"
        assert calls == [primary, primary, fallback]
        assert time.monotonic() - started_at < 0.2

    asyncio.run(run_test())


def test_fast_model_call_returns_normally():
    async def run_test():
        retry_middleware = ModelRetryMiddleware(timeout_seconds=0.01)

        async def handler(request):
            return ModelResponse(result=[AIMessage(content="success")])

        result = await retry_middleware.awrap_model_call(
            _request(FakeModel()), handler
        )

        assert result.result[0].content == "success"

    asyncio.run(run_test())
