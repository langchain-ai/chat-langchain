import asyncio
import time

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def test_timeout_falls_back_without_waiting_for_stalled_model():
    primary = object()
    fallback = object()
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback)

    async def handler(request):
        if request.model is primary:
            await asyncio.sleep(1)
        return AIMessage(content="fallback response")

    request = ModelRequest(
        model=primary,
        messages=[HumanMessage(content="Hello")],
    )
    started = time.monotonic()
    response = asyncio.run(
        fallback_middleware.awrap_model_call(
            request,
            lambda current_request: timeout_middleware.awrap_model_call(
                current_request, handler
            ),
        )
    )

    assert response.content == "fallback response"
    assert time.monotonic() - started < 0.5


def test_fast_model_call_passes_through():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.5)
    response = AIMessage(content="fast response")

    async def handler(_request):
        return response

    result = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(
                model=object(),
                messages=[HumanMessage(content="Hello")],
            ),
            handler,
        )
    )

    assert result is response
