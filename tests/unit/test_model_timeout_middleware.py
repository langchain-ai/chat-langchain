import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


class FakeChatModel(BaseChatModel):
    delay: float = 0
    response: str = "response"
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )


def test_model_timeout_middleware_raises_timeout_error():
    async def run_test():
        middleware = ModelTimeoutMiddleware(timeout_seconds=0.05)

        async def slow_handler(request):
            await asyncio.sleep(1)

        with pytest.raises(TimeoutError, match="0.05 seconds"):
            await middleware.awrap_model_call(None, slow_handler)

    asyncio.run(run_test())


def test_timeout_allows_fallback_model_to_respond():
    async def run_test():
        primary = FakeChatModel(delay=1)
        fallback = FakeChatModel(response="fallback response")
        agent = create_agent(
            model=primary,
            middleware=[
                ModelRetryMiddleware(max_retries=0),
                ModelFallbackMiddleware(fallback),
                ModelTimeoutMiddleware(timeout_seconds=0.05),
            ],
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "hello"}]}
        )

        assert result["messages"][-1].content == "fallback response"
        assert primary.calls == 1
        assert fallback.calls == 1

    asyncio.run(run_test())
