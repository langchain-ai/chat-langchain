"""Bound model calls with a configurable deadline."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelTimeoutMiddleware(AgentMiddleware):
    """Raise when an async model call exceeds the configured deadline."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run a model call with the configured deadline."""
        from src.agent.config import MODEL_TIMEOUT_SECONDS

        return await asyncio.wait_for(
            handler(request),
            timeout=MODEL_TIMEOUT_SECONDS,
        )


__all__ = ["ModelTimeoutMiddleware"]
