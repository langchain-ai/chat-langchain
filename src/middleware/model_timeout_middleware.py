"""Middleware for bounding model call duration."""

import asyncio
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise a timeout error when a model call exceeds its deadline."""

    def __init__(self, timeout_s: float):
        """Configure the model call timeout."""
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
        """Raise a descriptive error when an async model call times out."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError as error:
            model_name = getattr(request.model, "model_name", request.model)
            raise TimeoutError(
                f"Model call for {model_name} timed out after {self.timeout_s:g} seconds"
            ) from error


__all__ = ["ModelCallTimeoutMiddleware"]
