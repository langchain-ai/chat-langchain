"""Middleware for enforcing deadlines on asynchronous model calls."""

import asyncio
import os
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelTimeoutMiddleware(AgentMiddleware):
    """Enforce a timeout on each asynchronous model call."""

    def __init__(self, timeout_seconds: float | None = None):
        """Initialize the middleware with a configured timeout."""
        super().__init__()
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "45"))
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run the model call with an asynchronous timeout."""
        try:
            return await asyncio.wait_for(handler(request), self.timeout_seconds)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Model call timed out after {self.timeout_seconds} seconds"
            ) from exc

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Run synchronous model calls without applying an async timeout."""
        return handler(request)
