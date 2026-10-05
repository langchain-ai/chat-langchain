import asyncio
from typing import cast

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware


def _request(model: BaseChatModel) -> ModelRequest:
    return ModelRequest(model=model, messages=[HumanMessage(content="Hi")])


def test_model_call_timeout_raises_timeout_error():
    middleware = ModelTimeoutMiddleware(timeout_seconds=0.01)

    async def handler(_request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="late")])

    with pytest.raises(TimeoutError):
        asyncio.run(
            middleware.awrap_model_call(_request(cast(BaseChatModel, object())), handler)
        )


def test_timeout_allows_fallback_model_response():
    primary = cast(BaseChatModel, object())
    fallback = cast(BaseChatModel, object())
    timeout = ModelTimeoutMiddleware(timeout_seconds=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    calls: list[object] = []

    async def provider_handler(request: ModelRequest) -> ModelResponse:
        calls.append(request.model)
        if request.model is primary:
            await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="fallback response")])

    async def timed_handler(request: ModelRequest) -> ModelResponse:
        return await timeout.awrap_model_call(request, provider_handler)

    response = asyncio.run(
        fallback_middleware.awrap_model_call(_request(primary), timed_handler)
    )

    assert response.result[0].content == "fallback response"
    assert calls == [primary, fallback]
