"""Middleware for bounding individual model calls."""

import asyncio
import os
import queue
import threading
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

MODEL_CALL_TIMEOUT_SECONDS = float(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "30"))


class ModelTimeoutMiddleware(AgentMiddleware):
    """Bound each model call to a configurable number of seconds."""

    def __init__(self, timeout: float | None = None):
        """Initialize the model timeout."""
        super().__init__()
        self.timeout = MODEL_CALL_TIMEOUT_SECONDS if timeout is None else float(timeout)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Bound an asynchronous model call."""
        return await asyncio.wait_for(handler(request), timeout=self.timeout)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Bound a synchronous model call."""
        result_queue: queue.Queue[tuple[bool, ModelCallResult | BaseException]] = (
            queue.Queue(maxsize=1)
        )

        def run_handler() -> None:
            try:
                result_queue.put((True, handler(request)))
            except BaseException as error:
                result_queue.put((False, error))

        thread = threading.Thread(target=run_handler, daemon=True)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            raise TimeoutError(f"Model call exceeded {self.timeout:.2f} seconds")

        succeeded, result = result_queue.get()
        if not succeeded:
            raise result
        return result


__all__ = ["MODEL_CALL_TIMEOUT_SECONDS", "ModelTimeoutMiddleware"]
