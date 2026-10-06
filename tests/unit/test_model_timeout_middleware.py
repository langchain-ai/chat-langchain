import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="hello")])


def test_model_call_timeout_raises_timeout_error():
    async def run():
        middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

        async def handler(request):
            await asyncio.sleep(1)

        with pytest.raises(TimeoutError):
            await middleware.awrap_model_call(_request(object()), handler)

    asyncio.run(run())


def test_timeout_reaches_fallback_when_timeout_is_innermost():
    async def run():
        fallback_model = object()
        fallback = ModelFallbackMiddleware(fallback_model)
        timeout = ModelCallTimeoutMiddleware(timeout_s=0.01)
        calls = []

        async def handler(request):
            calls.append(request.model)
            if request.model is not fallback_model:
                await asyncio.sleep(1)
            return "fallback result"

        async def timed_handler(request):
            return await timeout.awrap_model_call(request, handler)

        result = await fallback.awrap_model_call(_request(object()), timed_handler)

        assert result == "fallback result"
        assert calls[-1] is fallback_model

    asyncio.run(run())
