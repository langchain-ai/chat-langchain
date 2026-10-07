"""Bound the duration of individual asynchronous model attempts."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise TimeoutError when an asynchronous model attempt stalls."""

    def __init__(self, timeout_s: float):
        """Set the time limit in seconds for each model attempt."""
        super().__init__()
        self.timeout_s = timeout_s

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Cancel an asynchronous model attempt that exceeds the time limit."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout_s)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        """Pass synchronous model calls through to the provider."""
        return handler(request)


__all__ = ["ModelCallTimeoutMiddleware"]
