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

DEFAULT_MODEL_CALL_TIMEOUT_SECONDS = 60.0


class ModelTimeoutMiddleware(AgentMiddleware):
    """Bound each model call to a configurable deadline."""

    def __init__(self, timeout_seconds: float | None = None):
        """Configure the model call timeout."""
        super().__init__()
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(
                os.getenv(
                    "MODEL_CALL_TIMEOUT_SECONDS",
                    str(DEFAULT_MODEL_CALL_TIMEOUT_SECONDS),
                )
            )
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Run synchronous model calls without changing their behavior."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Raise TimeoutError when an asynchronous model call exceeds its deadline."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout_seconds)


__all__ = ["ModelTimeoutMiddleware"]
