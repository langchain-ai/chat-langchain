"""Timeout middleware for individual model attempts."""

import asyncio
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelTimeoutMiddleware(AgentMiddleware):
    """Bound each individual model attempt."""

    def __init__(self, timeout_seconds: float):
        """Initialize the timeout middleware."""
        super().__init__()
        self.timeout_seconds = timeout_seconds

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run one model attempt with a deadline."""
        return await asyncio.wait_for(handler(request), self.timeout_seconds)


__all__ = ["ModelTimeoutMiddleware"]
