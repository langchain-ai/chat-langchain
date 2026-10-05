"""Middleware for bounding individual model calls."""

import asyncio
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Limit asynchronous model calls to a configured duration."""

    def __init__(self, timeout_s: float):
        """Initialize the middleware with a timeout in seconds."""
        super().__init__()
        self.timeout_s = timeout_s

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run a model call with a deadline."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout_s)


__all__ = ["ModelCallTimeoutMiddleware"]
