"""Tests for model timeout and fallback middleware wiring."""

import asyncio
import time

from langchain.agents.middleware import ModelFallbackMiddleware, ModelRequest
from langchain_core.messages import HumanMessage

from src.agent import config
from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


class FakeModel:
    """Fake model that either hangs or returns a response."""

    def __init__(self, response=None):
        self.response = response
        self.calls = 0

    async def ainvoke(self, messages):  # noqa: ARG002
        self.calls += 1
        if self.response is None:
            await asyncio.Event().wait()
        return self.response


def test_model_timeout_retries_then_falls_back(monkeypatch):
    """A stalled primary model exhausts retries before fallback runs."""
    monkeypatch.setattr(config, "MODEL_TIMEOUT_SECONDS", 0.01)

    primary = FakeModel()
    fallback = FakeModel("fallback response")
    timeout = ModelTimeoutMiddleware()
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    model_fallback = ModelFallbackMiddleware(fallback)
    request = ModelRequest(model=primary, messages=[HumanMessage(content="hello")])

    async def call_model(current_request):
        return await current_request.model.ainvoke(current_request.messages)

    async def call_with_timeout(current_request):
        return await timeout.awrap_model_call(current_request, call_model)

    async def call_with_retry(current_request):
        return await retry.awrap_model_call(current_request, call_with_timeout)

    async def call_with_fallback(current_request):
        return await model_fallback.awrap_model_call(current_request, call_with_retry)

    started = time.monotonic()
    result = asyncio.run(call_with_fallback(request))
    elapsed = time.monotonic() - started

    assert result == "fallback response"
    assert primary.calls == 2
    assert fallback.calls == 1
    assert elapsed < 0.2
