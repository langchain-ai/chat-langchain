import asyncio

import anyio
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_timeout_retries_then_falls_back():
    anyio.run(_assert_timeout_retries_then_falls_back)


async def _assert_timeout_retries_then_falls_back():
    primary_model = object()
    fallback_model = object()
    retry_middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        timeout=0.01,
    )
    fallback_middleware = ModelFallbackMiddleware(fallback_model)
    request = ModelRequest(model=primary_model, messages=[])
    attempts = []

    async def handler(model_request: ModelRequest) -> ModelResponse:
        attempts.append(model_request.model)
        if model_request.model is primary_model:
            await asyncio.Future()
        return ModelResponse(result=[AIMessage(content="fallback")])

    result = await fallback_middleware.awrap_model_call(
        request,
        lambda model_request: retry_middleware.awrap_model_call(
            model_request,
            handler,
        ),
    )

    assert result.result[0].content == "fallback"
    assert attempts == [primary_model, primary_model, fallback_model]
