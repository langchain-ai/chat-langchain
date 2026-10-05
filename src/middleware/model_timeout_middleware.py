"""Middleware for bounding individual model calls."""

import asyncio
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise a timeout when a model call exceeds its deadline."""

    def __init__(self, timeout_s: float = 60.0):
        """Initialize the middleware with a per-call timeout."""
        super().__init__()
        self.timeout_s = timeout_s

    def _model_name(self, request: ModelRequest) -> str | None:
        model = request.model
        if isinstance(model, str):
            return model
        return (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or getattr(model, "model_id", None)
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Bound an asynchronous model call to the configured timeout."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError as exc:
            model_name = self._model_name(request)
            detail = f" for model {model_name}" if model_name else ""
            raise TimeoutError(
                f"Model call timed out after {self.timeout_s:g} seconds{detail}"
            ) from exc

    def wrap_model_call(self, request, handler):
        """Pass synchronous model calls through without a deadline."""
        # Synchronous calls remain pass-through; the agent uses the async path.
        return handler(request)
