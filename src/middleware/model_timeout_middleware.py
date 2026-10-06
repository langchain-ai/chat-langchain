"""Middleware for enforcing model call deadlines."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Limit the duration of each asynchronous model call."""

    def __init__(self, timeout_s: float):
        """Initialize the model call timeout."""
        super().__init__()
        self.timeout_s = timeout_s

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Execute synchronous model calls without changing their behavior."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Execute an asynchronous model call with a deadline."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(f"model call exceeded {self.timeout_s}s") from exc


__all__ = ["ModelCallTimeoutMiddleware"]
