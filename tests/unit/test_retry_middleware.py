import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model: object) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hello")],
    )


def test_timeout_retries_each_model_attempt():
    attempts = 0

    async def handler(request: ModelRequest) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="never")])

    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout=0.01,
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(_request(object()), handler))

    assert attempts == 3


def test_timeout_retries_then_falls_back():
    primary_model = object()
    fallback_model = object()
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0, timeout=0.01)
    fallback = ModelFallbackMiddleware(fallback_model)
    attempts: list[object] = []

    async def model_handler(request: ModelRequest) -> ModelResponse:
        attempts.append(request.model)
        if request.model is primary_model:
            await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="fallback")])

    async def retry_handler(request: ModelRequest) -> ModelResponse:
        return await retry.awrap_model_call(request, model_handler)

    response = asyncio.run(
        fallback.awrap_model_call(
            _request(primary_model),
            retry_handler,
        )
    )

    assert response.result[0].content == "fallback"
    assert attempts == [primary_model, primary_model, fallback_model]
