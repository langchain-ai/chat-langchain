"""Tests for GracefulErrorMiddleware fallback recovery."""

import asyncio
import os

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.graceful_error_middleware import (
    DEFAULT_FALLBACK_MESSAGE,
    GracefulErrorMiddleware,
)
from src.middleware.retry_middleware import MalformedResponseError


class _BoomModel(FakeListChatModel):
    """Fake chat model that always raises MalformedResponseError."""

    async def _agenerate(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise MalformedResponseError("simulated malformed response")

    def _generate(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise MalformedResponseError("simulated malformed response")


def test_graceful_error_middleware_emits_fallback_message():
    """Graph should not propagate model exceptions; it should emit a fallback AIMessage."""
    model = _BoomModel(responses=["unused"])
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[GracefulErrorMiddleware()],
    )

    result = asyncio.run(
        agent.ainvoke({"messages": [HumanMessage(content="hi")]})
    )

    messages = result["messages"]
    assert any(isinstance(m, AIMessage) and m.content for m in messages), (
        "expected a non-empty AIMessage in final state"
    )
    last = messages[-1]
    assert isinstance(last, AIMessage)
    assert last.content == DEFAULT_FALLBACK_MESSAGE
