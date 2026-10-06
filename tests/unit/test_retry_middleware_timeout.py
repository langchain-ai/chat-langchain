"""Tests for model call timeout handling."""

import asyncio
import time

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


@pytest.mark.parametrize("max_retries", [0, 2])
def test_retry_middleware_times_out_hung_handler(max_retries):
    """A hung model call should be retried and then raise its timeout."""
    middleware = ModelRetryMiddleware(
        max_retries=max_retries,
        call_timeout=0.01,
        initial_delay=0,
    )
    attempts = 0

    async def handler(request):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    request = ModelRequest(model=None, messages=[HumanMessage(content="test")])
    started_at = time.monotonic()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(request, handler))

    assert attempts == max_retries + 1
    assert time.monotonic() - started_at < 1


class StubChatModel(BaseChatModel):
    """Minimal chat model used to exercise middleware composition."""

    label: str

    @property
    def _llm_type(self):
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        raise NotImplementedError


def test_fallback_handles_hung_primary_after_retry_timeout():
    """A timed-out primary should allow the fallback model to answer."""

    async def run_test():
        primary = StubChatModel(label="primary")
        fallback = StubChatModel(label="fallback")
        fallback_middleware = ModelFallbackMiddleware(fallback)
        retry_middleware = ModelRetryMiddleware(max_retries=0, call_timeout=0.01)
        request = ModelRequest(model=primary, messages=[])

        async def handler(model_request):
            async def invoke(request_to_model):
                if request_to_model.model is primary:
                    await asyncio.Event().wait()
                return ModelResponse(
                    result=[AIMessage(content=request_to_model.model.label)]
                )

            return await retry_middleware.awrap_model_call(model_request, invoke)

        return await fallback_middleware.awrap_model_call(request, handler)

    response = asyncio.run(run_test())

    assert response.result[0].content == "fallback"
