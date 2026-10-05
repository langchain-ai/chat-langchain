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

MODEL_CALL_TIMEOUT_SECONDS = float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "60"))


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Apply a deadline to asynchronous model calls."""

    def __init__(self, timeout_s: float = MODEL_CALL_TIMEOUT_SECONDS) -> None:
        """Initialize the model call timeout."""
        super().__init__()
        self.timeout_s = timeout_s

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
        """Enforce the timeout for an asynchronous model call."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
