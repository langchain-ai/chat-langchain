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
    """Bound each individual model call by a configurable timeout."""

    def __init__(self, timeout: float):
        """Initialize the timeout middleware."""
        super().__init__()
        self.timeout = timeout

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Raise when the wrapped model call exceeds the timeout."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout)


__all__ = ["ModelCallTimeoutMiddleware"]
