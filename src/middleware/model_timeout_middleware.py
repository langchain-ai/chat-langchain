"""Middleware for bounding individual model calls."""

import asyncio
import os
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

MODEL_CALL_TIMEOUT_SECONDS = float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "30"))


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise a failure when a model call exceeds its configured deadline."""

    def __init__(self, timeout_s: float | None = None) -> None:
        """Initialize the middleware with an optional timeout override."""
        self.timeout_s = MODEL_CALL_TIMEOUT_SECONDS if timeout_s is None else timeout_s

    def wrap_model_call(self, request, handler):
        """Pass synchronous model calls through unchanged."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Bound an asynchronous model call and convert timeouts to failures."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError as exc:
            model_name = (
                getattr(request.model, "model", None)
                or getattr(request.model, "model_name", None)
                or request.model.__class__.__name__
            )
            raise TimeoutError(
                f"Model call for {model_name} timed out after {self.timeout_s:g} seconds"
            ) from exc


__all__ = ["MODEL_CALL_TIMEOUT_SECONDS", "ModelCallTimeoutMiddleware"]
