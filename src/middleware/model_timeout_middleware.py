"""Middleware for bounding individual model calls."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise when an individual model call exceeds its deadline."""

    def __init__(self, timeout_s: float):
        """Initialize the model call timeout."""
        super().__init__()
        self.timeout_s = timeout_s

    def wrap_model_call(self, request: ModelRequest, handler: Callable):
        """Pass synchronous model calls through unchanged."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Bound an asynchronous model call by the configured timeout."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
