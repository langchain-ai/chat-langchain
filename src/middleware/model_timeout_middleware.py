"""Timeout middleware for model calls."""

import asyncio
import os
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Bound model calls using the configured timeout."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Delegate synchronous model execution."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Raise when asynchronous model execution exceeds the timeout."""
        return await asyncio.wait_for(
            handler(request),
            timeout=float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "45")),
        )


__all__ = ["ModelCallTimeoutMiddleware"]
