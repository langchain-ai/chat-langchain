import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


class DelayedFakeChatModel(FakeChatModel):
    response: str
    delay: float = 0

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        await asyncio.sleep(self.delay)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="Hello")])


def test_model_call_timeout_raises_timeout_error():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.02)
    model = DelayedFakeChatModel(response="too late", delay=0.1)

    async def handler(request):
        return await request.model.ainvoke(request.messages)

    with pytest.raises(TimeoutError, match="model call exceeded 0.02s"):
        asyncio.run(middleware.awrap_model_call(_request(model), handler))


def test_model_call_timeout_allows_fallback_to_finish():
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    fallback_middleware = ModelFallbackMiddleware(
        DelayedFakeChatModel(response="fallback")
    )
    primary = DelayedFakeChatModel(response="primary", delay=0.1)

    async def handler(request):
        return await request.model.ainvoke(request.messages)

    async def timed_handler(request):
        return await timeout_middleware.awrap_model_call(request, handler)

    result = asyncio.run(
        fallback_middleware.awrap_model_call(_request(primary), timed_handler)
    )

    assert result.content == "fallback"
