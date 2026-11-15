"""Experiment: run every docs-agent model call on a configured chat model instance
(e.g. claude-haiku-5-5 with thinking disabled). v2 eval only, not for merge."""

import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class ModelOverrideMiddleware(AgentMiddleware):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        return handler(request.override(model=self.model))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]):
        return await handler(request.override(model=self.model))


def build_haiku_nothink() -> Any:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model="claude-haiku-5-5", thinking={"type": "enabled", "budget_tokens": 1024}, max_tokens=16000)
