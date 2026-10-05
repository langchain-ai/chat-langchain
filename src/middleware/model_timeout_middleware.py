"""Middleware for bounding model call duration."""

import asyncio
import logging
import os
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)

MODEL_CALL_TIMEOUT_SECONDS = float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "60.0"))


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise when a model call exceeds its configured duration."""

    def __init__(self, timeout_seconds: float = MODEL_CALL_TIMEOUT_SECONDS):
        """Initialize the middleware with a timeout in seconds."""
        super().__init__()
        self.timeout_seconds = timeout_seconds

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Bound one model call and re-raise timeout failures."""
        try:
            return await asyncio.wait_for(
                handler(request), timeout=self.timeout_seconds
            )
        except TimeoutError:
            model_name = (
                getattr(request.model, "model_name", None)
                or getattr(request.model, "model", None)
                or type(request.model).__name__
            )
            logger.warning(
                "Model call timed out for %s after %.1f seconds",
                model_name,
                self.timeout_seconds,
            )
            raise
