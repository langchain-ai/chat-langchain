"""Timeout middleware for model calls."""

import asyncio
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelTimeoutMiddleware(AgentMiddleware):
    """Limit the time spent on each model call."""

    def __init__(self, timeout: float):
        """Initialize timeout middleware."""
        super().__init__()
        self.timeout = timeout

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Pass through synchronous model calls."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Raise TimeoutError when an asynchronous model call exceeds the deadline."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout)
        except TimeoutError as error:
            raise TimeoutError(f"model call exceeded {self.timeout}s") from error


__all__ = ["ModelTimeoutMiddleware"]
