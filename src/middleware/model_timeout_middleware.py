"""Middleware for bounding asynchronous model call duration."""

import asyncio
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Enforce a deadline for each asynchronous model call."""

    def __init__(self, timeout_s: float):
        """Initialize the middleware with a timeout in seconds."""
        super().__init__()
        self.timeout_s = timeout_s

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Pass synchronous model calls through unchanged."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Raise a timeout when an asynchronous model call exceeds the deadline."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError:
            logger.warning("Model call exceeded %.2f seconds", self.timeout_s)
            raise TimeoutError(f"model call exceeded {self.timeout_s}s") from None


__all__ = ["ModelCallTimeoutMiddleware"]
