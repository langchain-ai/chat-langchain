"""Tests for model call timeout and fallback behavior."""

import asyncio

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware import retry_middleware as retry_module
from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model: object) -> ModelRequest:
    return ModelRequest(model=model, messages=[HumanMessage(content="Hello")])


def test_timed_out_model_retries_then_falls_back(monkeypatch):
    """A timed-out primary model retries before fallback handles the error."""
    monkeypatch.setattr(retry_module, "MODEL_CALL_TIMEOUT_SECONDS", 0.01)
    primary_model = object()
    fallback_model = object()
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    fallback = ModelFallbackMiddleware(fallback_model)
    calls = {"primary": 0, "fallback": 0}

    async def handler(request):
        if request.model is primary_model:
            calls["primary"] += 1
            await asyncio.Event().wait()
        calls["fallback"] += 1
        return ModelResponse([AIMessage(content="fallback response")])

    async def wrapped_handler(request):
        if request.model is primary_model:
            return await retry.awrap_model_call(request, handler)
        return await handler(request)

    result = asyncio.run(
        fallback.awrap_model_call(_request(primary_model), wrapped_handler)
    )

    assert result.result[0].content == "fallback response"
    assert calls == {"primary": 2, "fallback": 1}


def test_fast_model_call_is_not_delayed(monkeypatch):
    """A model response that arrives before the deadline returns normally."""
    monkeypatch.setattr(retry_module, "MODEL_CALL_TIMEOUT_SECONDS", 0.01)
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)

    async def handler(request):  # noqa: ARG001
        return ModelResponse([AIMessage(content="fast response")])

    result = asyncio.run(retry.awrap_model_call(_request(object()), handler))

    assert result.result[0].content == "fast response"
