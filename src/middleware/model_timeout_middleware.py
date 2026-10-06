"""Middleware for enforcing deadlines around model calls."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise TimeoutError when a model call exceeds its deadline."""

    def __init__(self, timeout_s: float | None = None):
        """Initialize the middleware with a per-call timeout."""
        super().__init__()
        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "30"))
        )
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")

    def _timeout_error(self) -> TimeoutError:
        return TimeoutError(
            f"Model call exceeded the {self.timeout_s:g}-second deadline"
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Run a synchronous model call with a deadline."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, request)
        try:
            return future.result(timeout=self.timeout_s)
        except FutureTimeoutError:
            future.cancel()
            raise self._timeout_error() from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run an asynchronous model call with a deadline."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout_s)
        except TimeoutError:
            raise self._timeout_error() from None


__all__ = ["ModelCallTimeoutMiddleware"]
