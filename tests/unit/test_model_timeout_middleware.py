"""Tests for model call timeout and fallback behavior."""

import asyncio
from typing import cast

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel

from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware


class FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name


def _request(model: FakeModel) -> ModelRequest:
    return ModelRequest(model=cast(BaseChatModel, model), messages=[])


def test_model_timeout_middleware_raises_timeout_error():
    middleware = ModelTimeoutMiddleware(timeout_seconds=0.01)

    async def run():
        async def handler(request):  # noqa: ARG001
            await asyncio.sleep(1)
            return ModelResponse(result=[])

        return await middleware.awrap_model_call(
            _request(FakeModel("primary")), handler
        )

    with pytest.raises(TimeoutError):
        asyncio.run(run())


def test_model_fallback_handles_timeout_from_inner_middleware():
    primary = FakeModel("primary")
    fallback = FakeModel("fallback")
    timeout = ModelTimeoutMiddleware(timeout_seconds=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    calls = []

    async def run():
        async def handler(request):
            calls.append(request.model.model_name)

            async def model_handler(inner_request):
                if inner_request.model is primary:
                    await asyncio.sleep(1)
                return ModelResponse(result=[])

            return await timeout.awrap_model_call(request, model_handler)

        return await fallback_middleware.awrap_model_call(_request(primary), handler)

    result = asyncio.run(run())

    assert result.result == []
    assert calls == ["primary", "fallback"]
