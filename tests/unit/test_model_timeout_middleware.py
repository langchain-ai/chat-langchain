import asyncio
import time

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.middleware.retry_middleware import ModelRetryMiddleware, ModelTimeoutMiddleware


class StubModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="stub"))]
        )


def test_model_timeout_middleware_raises_for_never_resolving_handler():
    async def run():
        middleware = ModelTimeoutMiddleware(timeout=0.01)
        request = ModelRequest(model=StubModel(), messages=[])

        async def never_resolves(request):
            await asyncio.Event().wait()

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await middleware.awrap_model_call(request, never_resolves)

        assert time.monotonic() - started < 0.5

    asyncio.run(run())


def test_timeout_reaches_fallback_without_retrying():
    async def run():
        primary = StubModel()
        fallback_model = StubModel()
        timeout_middleware = ModelTimeoutMiddleware(timeout=0.01)
        retry_middleware = ModelRetryMiddleware(
            max_retries=2,
            initial_delay=1,
        )
        fallback_middleware = ModelFallbackMiddleware(fallback_model)
        request = ModelRequest(model=primary, messages=[])
        attempts = []

        async def handler(request):
            attempts.append(request.model)
            if request.model is primary:
                await asyncio.Event().wait()
            return ModelResponse(result=[AIMessage(content="fallback")])

        async def retry_handler(request):
            return await retry_middleware.awrap_model_call(
                request,
                lambda nested_request: timeout_middleware.awrap_model_call(
                    nested_request, handler
                ),
            )

        started = time.monotonic()
        response = await fallback_middleware.awrap_model_call(request, retry_handler)

        assert response.result[0].content == "fallback"
        assert attempts == [primary, fallback_model]
        assert time.monotonic() - started < 0.5

    asyncio.run(run())
