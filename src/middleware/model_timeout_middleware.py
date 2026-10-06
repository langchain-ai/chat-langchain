"""Timeout middleware for asynchronous model calls."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Limit the time allowed for each asynchronous model call."""

    def __init__(self, timeout: float):
        """Initialize the model call timeout."""
        super().__init__()
        self.timeout = timeout

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Raise TimeoutError when the model call exceeds the deadline."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout)


__all__ = ["ModelCallTimeoutMiddleware"]
