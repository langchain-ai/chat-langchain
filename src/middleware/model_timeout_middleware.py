"""Middleware for enforcing model call deadlines."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise when a model call exceeds its configured deadline."""

    def __init__(self, timeout_seconds: float):
        """Initialize the middleware with a timeout in seconds."""
        super().__init__()
        self.timeout_seconds = timeout_seconds

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply the deadline to an asynchronous model call."""
        try:
            return await asyncio.wait_for(
                handler(request), timeout=self.timeout_seconds
            )
        except TimeoutError as exc:
            raise TimeoutError(f"model call exceeded {self.timeout_seconds}s") from exc

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply the deadline to a synchronous model call."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, request)
        try:
            return future.result(timeout=self.timeout_seconds)
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"model call exceeded {self.timeout_seconds}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["ModelCallTimeoutMiddleware"]
