import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.model_call_timeout_middleware import ModelCallTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def _request():
    return ModelRequest(
        model=SimpleNamespace(name="primary"),
        messages=[HumanMessage(content="hello")],
    )


def test_model_call_timeout_raises_near_configured_deadline():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

    async def handler(request):  # noqa: ARG001
        await asyncio.sleep(1)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(_request(), handler))


def test_model_call_timeout_allows_fallback_response():
    timeout = ModelCallTimeoutMiddleware(timeout_s=0.01)
    fallback = ModelFallbackMiddleware.__new__(ModelFallbackMiddleware)
    fallback.models = [SimpleNamespace(name="fallback")]

    async def handler(request):
        if request.model.name == "primary":
            await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="fallback response")])

    async def call_with_timeout(request):
        return await timeout.awrap_model_call(request, handler)

    result = asyncio.run(fallback.awrap_model_call(_request(), call_with_timeout))

    assert result.result[0].content == "fallback response"


def test_model_retry_does_not_retry_timeout():
    middleware = ModelRetryMiddleware(max_retries=2)
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise asyncio.TimeoutError

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert calls == 1
