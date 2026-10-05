"""Tests for model call timeouts and exhausted timeout handling."""

import asyncio

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

import src.middleware.model_timeout_middleware as timeout_module
from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model: object = object()) -> ModelRequest:
    return ModelRequest(model=model, messages=[HumanMessage(content="Hello")])


def test_model_timeout_middleware_raises_after_timeout(monkeypatch):
    monkeypatch.setattr(timeout_module, "MODEL_CALL_TIMEOUT_SECONDS", 0.01)
    middleware = ModelTimeoutMiddleware()

    async def handler(_request):
        await asyncio.sleep(1)

    try:
        asyncio.run(middleware.awrap_model_call(_request(), handler))
    except TimeoutError:
        pass
    else:
        raise AssertionError("model timeout did not raise TimeoutError")


def test_model_fallback_runs_after_primary_timeout(monkeypatch):
    monkeypatch.setattr(timeout_module, "MODEL_CALL_TIMEOUT_SECONDS", 0.01)
    primary_model = object()
    fallback_model = object()
    fallback = ModelFallbackMiddleware(fallback_model)
    timeout = ModelTimeoutMiddleware()
    calls = []

    async def model_handler(request):
        calls.append(request.model)
        if request.model is primary_model:
            await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="fallback answer")])

    async def wrapped_handler(request):
        return await timeout.awrap_model_call(request, model_handler)

    response = asyncio.run(
        fallback.awrap_model_call(_request(primary_model), wrapped_handler)
    )

    assert response.result[0].content == "fallback answer"
    assert calls == [primary_model, fallback_model]


def test_exhausted_model_timeouts_return_apology():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        raise TimeoutError

    response = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert calls == 3
    assert "took too long" in response.result[0].content
