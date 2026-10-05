"""Middleware for bounding individual model call durations."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class ModelTimeoutMiddleware(AgentMiddleware):
    """Limit the duration of each model call."""

    def __init__(self, timeout_seconds: float | None = None):
        """Initialize the middleware with an environment-configurable deadline."""
        super().__init__()
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "45"))
        )

    def _model_id(self, request: ModelRequest) -> str:
        model = request.model
        return (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or getattr(model, "model_id", None)
            or (model if isinstance(model, str) else "unknown")
        )

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
        """Limit asynchronous model calls and re-raise timeout failures."""
        try:
            return await asyncio.wait_for(
                handler(request), timeout=self.timeout_seconds
            )
        except TimeoutError as exc:
            logger.warning(
                "Model call timed out after %.1fs (model=%s)",
                self.timeout_seconds,
                self._model_id(request),
            )
            raise TimeoutError(
                f"Model call timed out after {self.timeout_seconds:.1f}s"
            ) from exc


__all__ = ["ModelTimeoutMiddleware"]
