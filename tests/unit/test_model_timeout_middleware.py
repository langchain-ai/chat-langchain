"""Tests for per-attempt model call timeouts."""

import asyncio

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def test_model_call_timeout_raises_timeout_error():
    async def run_test() -> None:
        middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

        async def handler(request: ModelRequest) -> ModelResponse:
            await asyncio.sleep(1)
            return ModelResponse(result=[])

        try:
            await middleware.awrap_model_call(
                ModelRequest(model="primary", messages=[]), handler
            )
        except TimeoutError as error:
            assert str(error) == "model call exceeded 0.01s"
        else:
            raise AssertionError("Expected model call timeout")

    asyncio.run(run_test())


def test_model_call_timeout_allows_fallback_attempt():
    async def run_test() -> None:
        timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
        fallback_model = object()
        fallback_middleware = ModelFallbackMiddleware(fallback_model)
        attempts: list[object] = []

        async def handler(request: ModelRequest) -> ModelResponse:
            attempts.append(request.model)

            async def invoke(_: ModelRequest) -> ModelResponse:
                if request.model == "primary":
                    await asyncio.sleep(1)
                return ModelResponse(result=[])

            return await timeout_middleware.awrap_model_call(request, invoke)

        response = await fallback_middleware.awrap_model_call(
            ModelRequest(model="primary", messages=[]), handler
        )

        assert isinstance(response, ModelResponse)
        assert attempts == ["primary", fallback_model]

    asyncio.run(run_test())
