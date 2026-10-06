import asyncio

import anyio
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_hung_model_retries_then_uses_fallback():
    """A hung model attempt is bounded before fallback is selected."""

    async def run_test():
        primary = FakeListChatModel(responses=["unused"])
        fallback = FakeListChatModel(responses=["fallback response"])
        timeout = ModelTimeoutMiddleware(timeout_seconds=0.01)
        retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)
        model_fallback = ModelFallbackMiddleware(fallback)
        request = ModelRequest(
            model=primary,
            messages=[HumanMessage(content="hello")],
        )
        attempts = 0

        async def model_handler(current_request):
            nonlocal attempts
            if current_request.model is primary:
                attempts += 1
                await asyncio.Event().wait()
            return await current_request.model.ainvoke(current_request.messages)

        result = await model_fallback.awrap_model_call(
            request,
            lambda current_request: retry.awrap_model_call(
                current_request,
                lambda retry_request: timeout.awrap_model_call(
                    retry_request, model_handler
                ),
            ),
        )

        assert result.content == "fallback response"
        assert attempts == 2

    anyio.run(run_test)


def test_model_timeout_does_not_affect_fast_model():
    """A model completing before the deadline returns normally."""

    async def run_test():
        model = FakeListChatModel(responses=["normal response"])
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="hello")],
        )

        result = await ModelTimeoutMiddleware(timeout_seconds=0.1).awrap_model_call(
            request,
            lambda current_request: current_request.model.ainvoke(
                current_request.messages
            ),
        )

        assert result.content == "normal response"

    anyio.run(run_test)
