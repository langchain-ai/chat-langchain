import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def test_fast_model_call_passes_through():
    async def run_test():
        response = ModelResponse(result=[])
        middleware = ModelCallTimeoutMiddleware(timeout_seconds=0.05)

        async def handler(request):
            return response

        assert await middleware.awrap_model_call(None, handler) is response

    asyncio.run(run_test())


def test_stalled_model_call_raises_timeout():
    async def run_test():
        middleware = ModelCallTimeoutMiddleware(timeout_seconds=0.01)

        async def handler(request):
            await asyncio.sleep(1)
            return ModelResponse(result=[])

        with pytest.raises(TimeoutError, match="model call exceeded 0.01s"):
            await middleware.awrap_model_call(None, handler)

    asyncio.run(run_test())


def test_timeout_allows_fallback_model_to_answer():
    async def run_test():
        primary = FakeListChatModel(responses=["primary"])
        fallback = FakeListChatModel(responses=["fallback"])
        fallback_middleware = ModelFallbackMiddleware(fallback)
        timeout_middleware = ModelCallTimeoutMiddleware(timeout_seconds=0.01)
        request = ModelRequest(model=primary, messages=[HumanMessage("hello")])

        async def handler(current_request):
            if current_request.model is primary:
                await asyncio.sleep(1)
            return ModelResponse(
                result=[await current_request.model.ainvoke(current_request.messages)]
            )

        response = await fallback_middleware.awrap_model_call(
            request,
            lambda current_request: timeout_middleware.awrap_model_call(
                current_request, handler
            ),
        )

        assert response.result[0].content == "fallback"

    asyncio.run(run_test())
