"""Bound model calls with a per-attempt timeout."""

import asyncio
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise when a model call exceeds its configured timeout."""

    def __init__(self, timeout_s: float):
        """Initialize the timeout middleware."""
        super().__init__()
        self.timeout_s = timeout_s

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run a model call with a hard deadline."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError as error:
            raise TimeoutError(f"model call exceeded {self.timeout_s}s") from error


__all__ = ["ModelCallTimeoutMiddleware"]
