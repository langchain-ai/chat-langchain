"""Middleware for enforcing per-call model deadlines."""

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
    """Raise when a model call exceeds its configured deadline."""

    def __init__(self, timeout_s: float):
        """Initialize the model call timeout."""
        super().__init__()
        self.timeout_s = timeout_s

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Run a synchronous model call without a timeout wrapper."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run an asynchronous model call with a timeout wrapper."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError as exc:
            logger.warning("Model call timed out after %ss", self.timeout_s)
            raise TimeoutError(f"model call exceeded {self.timeout_s}s") from exc


__all__ = ["ModelCallTimeoutMiddleware"]
