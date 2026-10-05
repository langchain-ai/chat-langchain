"""Bound model calls to a configurable timeout."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

from src.agent.config import MODEL_CALL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class ModelTimeoutMiddleware(AgentMiddleware):
    """Enforce a timeout for asynchronous model calls."""

    def wrap_model_call(self, request, handler):
        """Pass through synchronous model calls."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Raise TimeoutError when an asynchronous model call stalls."""
        try:
            return await asyncio.wait_for(
                handler(request), timeout=MODEL_CALL_TIMEOUT_SECONDS
            )
        except TimeoutError:
            logger.warning(
                "Model call timed out after %.2f seconds",
                MODEL_CALL_TIMEOUT_SECONDS,
            )
            raise


__all__ = ["ModelTimeoutMiddleware"]
