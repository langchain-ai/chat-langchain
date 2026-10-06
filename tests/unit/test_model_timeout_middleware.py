"""Tests for model call timeout behavior."""

import asyncio

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


class FakeModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return None


def test_stalled_model_is_retried_then_fallback_runs(monkeypatch):
    monkeypatch.setenv("MODEL_CALL_TIMEOUT_SECONDS", "0.01")
    primary = FakeModel()
    fallback = FakeModel()
    calls = {"primary": 0, "fallback": 0}
    timeout = ModelCallTimeoutMiddleware()
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    model_fallback = ModelFallbackMiddleware(fallback)
    request = ModelRequest(model=primary, messages=[])

    async def model_handler(current_request):
        model_name = "primary" if current_request.model is primary else "fallback"
        calls[model_name] += 1
        if current_request.model is primary:
            await asyncio.sleep(10)
        return ModelResponse(result=[AIMessage(content="fallback")])

    async def retry_handler(current_request):
        return await timeout.awrap_model_call(current_request, model_handler)

    async def fallback_handler(current_request):
        return await retry.awrap_model_call(current_request, retry_handler)

    result = asyncio.run(model_fallback.awrap_model_call(request, fallback_handler))

    assert result.result[0].content == "fallback"
    assert calls["primary"] == 2
    assert calls["fallback"] == 1
