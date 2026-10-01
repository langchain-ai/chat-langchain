"""Circuit breaker for permanent primary model credential failures."""

import logging
import os
import time
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from src.agent.config import is_credential_error

logger = logging.getLogger(__name__)


class PrimaryCircuitBreakerMiddleware(AgentMiddleware):
    """Skip an auth-failing primary model during a cooldown window."""

    def __init__(
        self,
        primary_model: str,
        fallback_model: str,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Initialize the process-local primary circuit breaker."""
        super().__init__()
        self.primary_model_name = primary_model.split(":", 1)[-1]
        self.fallback_model = init_chat_model(model=fallback_model)
        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else float(os.getenv("PRIMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "300"))
        )
        self._failure_at: float | None = None
        self._warning_emitted = False

    def _is_primary(self, request: ModelRequest) -> bool:
        model_name = getattr(request.model, "model", None) or getattr(
            request.model, "model_name", None
        )
        return model_name == self.primary_model_name

    def _circuit_open(self) -> bool:
        return self._failure_at is not None and (
            time.monotonic() - self._failure_at < self.cooldown_seconds
        )

    def _mark_fallback_metadata(self) -> None:
        try:
            from langgraph.config import get_config

            metadata = get_config().setdefault("metadata", {})
            metadata.update(
                served_by_fallback=True,
                primary_failure_reason="auth",
            )
        except RuntimeError:
            return

    def _record_failure(self) -> None:
        if self._failure_at is None or not self._circuit_open():
            self._failure_at = time.monotonic()
            self._warning_emitted = False
        if not self._warning_emitted:
            logger.warning(
                "Primary model credential failure; opening circuit breaker",
                extra={"primary_failure_reason": "auth"},
            )
            self._warning_emitted = True

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        """Route synchronous primary calls through the breaker."""
        if not self._is_primary(request):
            if self._circuit_open():
                self._mark_fallback_metadata()
            return handler(request)
        if self._circuit_open():
            self._mark_fallback_metadata()
            return handler(request.override(model=self.fallback_model))
        try:
            return handler(request)
        except Exception as exc:
            if is_credential_error(exc):
                self._record_failure()
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse | AIMessage]],
    ) -> ModelResponse | AIMessage:
        """Route asynchronous primary calls through the breaker."""
        if not self._is_primary(request):
            if self._circuit_open():
                self._mark_fallback_metadata()
            return await handler(request)
        if self._circuit_open():
            self._mark_fallback_metadata()
            return await handler(request.override(model=self.fallback_model))
        try:
            return await handler(request)
        except Exception as exc:
            if is_credential_error(exc):
                self._record_failure()
            raise
