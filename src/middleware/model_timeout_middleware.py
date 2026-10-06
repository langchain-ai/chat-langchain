"""Bound asynchronous model calls so retries and fallbacks can run."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelTimeoutMiddleware(AgentMiddleware):
    """Raise a clear timeout when an asynchronous model call exceeds its bound."""

    def __init__(self, timeout_seconds: float):
        """Initialize the model timeout in seconds."""
        super().__init__()
        self.timeout_seconds = timeout_seconds

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Bound one asynchronous model call and raise a clear timeout."""
        try:
            return await asyncio.wait_for(handler(request), self.timeout_seconds)
        except TimeoutError as error:
            raise TimeoutError(
                f"Model call timed out after {self.timeout_seconds:g} seconds"
            ) from error

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Use the handler directly for synchronous calls."""
        return handler(request)


__all__ = ["ModelTimeoutMiddleware"]
