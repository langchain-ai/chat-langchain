"""Middleware for bounding individual model calls."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise when a model call exceeds its deadline."""

    def __init__(self, timeout_seconds: float):
        """Initialize the model call timeout."""
        super().__init__()
        self.timeout_seconds = timeout_seconds

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
        """Bound an asynchronous model call to the configured timeout."""
        try:
            return await asyncio.wait_for(
                handler(request), timeout=self.timeout_seconds
            )
        except TimeoutError as exc:
            logger.error("Model call exceeded %s seconds", self.timeout_seconds)
            raise TimeoutError(f"Model call exceeded {self.timeout_seconds}s") from exc


__all__ = ["ModelCallTimeoutMiddleware"]
