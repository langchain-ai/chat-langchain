"""Timeout middleware for model calls."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse


class ModelTimeoutMiddleware(AgentMiddleware):
    """Limit each model call to a configured number of seconds."""

    def __init__(self, timeout: float):
        """Initialize timeout middleware."""
        super().__init__()
        self.timeout = timeout

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap a model call with a timeout."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout)
