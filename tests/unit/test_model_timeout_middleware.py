import asyncio
import os
import time

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import ConfigDict

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from agent import docs_agent_middleware
from src.agent import config
from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


class StubModel(BaseChatModel):
    model_name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError


def _request(model: BaseChatModel) -> ModelRequest:
    return ModelRequest(model=model, messages=[])


def test_hanging_primary_times_out_then_fallback_succeeds():
    asyncio.run(_test_hanging_primary_times_out_then_fallback_succeeds())


async def _test_hanging_primary_times_out_then_fallback_succeeds():
    primary = StubModel(model_name="primary")
    fallback = StubModel(model_name="fallback")
    timeout = ModelTimeoutMiddleware(timeout_seconds=0.01)
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    calls = {"primary": 0, "fallback": 0}

    async def model_handler(request):
        calls[request.model.model_name] += 1
        if request.model is primary:
            await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="ok")])

    async def bounded_handler(request):
        return await timeout.awrap_model_call(request, model_handler)

    async def retrying_handler(request):
        return await retry.awrap_model_call(request, bounded_handler)

    started = time.monotonic()
    result = await fallback_middleware.awrap_model_call(
        _request(primary), retrying_handler
    )
    elapsed = time.monotonic() - started

    assert result.result[0].content == "ok"
    assert calls == {"primary": 2, "fallback": 1}
    assert elapsed < 0.1


def test_fast_primary_is_not_delayed():
    asyncio.run(_test_fast_primary_is_not_delayed())


async def _test_fast_primary_is_not_delayed():
    timeout = ModelTimeoutMiddleware(timeout_seconds=0.01)

    async def handler(request):
        return ModelResponse(result=[AIMessage(content="ok")])

    started = time.monotonic()
    result = await timeout.awrap_model_call(None, handler)

    assert result.result[0].content == "ok"
    assert time.monotonic() - started < 0.01


def test_timeout_raises_builtin_timeout_error_with_message():
    asyncio.run(_test_timeout_raises_builtin_timeout_error_with_message())


async def _test_timeout_raises_builtin_timeout_error_with_message():
    timeout = ModelTimeoutMiddleware(timeout_seconds=0.01)

    async def handler(request):
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError, match="timed out after 0.01 seconds"):
        await timeout.awrap_model_call(None, handler)


def test_timeout_middleware_is_inside_retry_and_fallback():
    types = [type(m).__name__ for m in docs_agent_middleware]

    assert types.index("ModelTimeoutMiddleware") < types.index("ModelRetryMiddleware")
    assert types.index("ModelTimeoutMiddleware") < types.index(
        "ModelFallbackMiddleware"
    )
    assert config.MODEL_CALL_TIMEOUT_SECONDS == 45
