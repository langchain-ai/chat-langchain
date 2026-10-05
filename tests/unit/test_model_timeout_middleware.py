import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


class FakeChatModel(BaseChatModel):
    model_name: str
    response: str | None = None
    wait_forever: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response or ""))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.wait_forever:
            await asyncio.Event().wait()
        return self._generate(messages, stop, run_manager, **kwargs)


async def _invoke_with_timeout(middleware, request, timeout_middleware):
    async def handler(current_request):
        return await timeout_middleware.awrap_model_call(
            current_request,
            lambda request: request.model.ainvoke(request.messages),
        )

    return await middleware.awrap_model_call(request, handler)


def test_model_call_timeout_raises_for_stalled_model():
    model = FakeChatModel(model_name="stalled", wait_forever=True)
    request = ModelRequest(model=model, messages=[HumanMessage(content="hello")])

    async def run():
        timeout_middleware = ModelCallTimeoutMiddleware(timeout=0.01)
        return await timeout_middleware.awrap_model_call(
            request,
            lambda current_request: current_request.model.ainvoke(
                current_request.messages
            ),
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())


def test_model_call_timeout_allows_fallback_model_to_answer():
    primary = FakeChatModel(model_name="stalled", wait_forever=True)
    fallback = FakeChatModel(model_name="fallback", response="fallback answer")
    request = ModelRequest(model=primary, messages=[HumanMessage(content="hello")])

    result = asyncio.run(
        _invoke_with_timeout(
            ModelFallbackMiddleware(fallback),
            request,
            ModelCallTimeoutMiddleware(timeout=0.01),
        )
    )

    assert result.content == "fallback answer"


def test_model_retry_retries_timeout_errors():
    model = FakeChatModel(model_name="primary")
    request = ModelRequest(model=model, messages=[HumanMessage(content="hello")])
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise asyncio.TimeoutError
        return ModelResponse(result=[AIMessage(content="answer")])

    result = asyncio.run(
        ModelRetryMiddleware(max_retries=1, initial_delay=0).awrap_model_call(
            request, handler
        )
    )

    assert result.result[0].content == "answer"
    assert attempts == 2
