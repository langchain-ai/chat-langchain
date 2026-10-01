"""Fallback middleware for provider authentication failures."""

import logging
import threading
import time
from typing import Awaitable, Callable

import langsmith as ls
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import _sanitize_request_for_fallback
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

from src.middleware.retry_middleware import is_authentication_error

logger = logging.getLogger(__name__)


def _model_id(model: object) -> str | None:
    return getattr(model, "model_name", None) or getattr(model, "model", None)


def _error_reason(exception: BaseException) -> str:
    for attribute in ("reason", "code", "status_code", "http_status"):
        value = getattr(exception, attribute, None)
        if value:
            return f"{type(exception).__name__}: {value}"
    return f"{type(exception).__name__}: {exception}"


class AuthenticationAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Skip an unauthenticated primary model during a cooldown window."""

    def __init__(
        self,
        primary_model_id: str,
        *additional_models: str,
        cooldown_seconds: float = 300,
    ) -> None:
        """Configure the primary model and its fallback cooldown."""
        super().__init__(*additional_models)
        self.primary_model_id = primary_model_id
        self.cooldown_seconds = cooldown_seconds
        self._cooldown_until = 0.0
        self._primary_error = ""
        self._lock = threading.Lock()

    def _primary_is_active(self, request: ModelRequest) -> bool:
        model_id = _model_id(request.model)
        return model_id in {
            self.primary_model_id,
            self.primary_model_id.split(":", maxsplit=1)[-1],
        }

    def _cooldown_state(self) -> tuple[bool, str]:
        with self._lock:
            return time.monotonic() < self._cooldown_until, self._primary_error

    def _record_auth_failure(self, exception: BaseException) -> None:
        reason = _error_reason(exception)
        should_log = False
        with self._lock:
            if time.monotonic() >= self._cooldown_until:
                should_log = True
            self._cooldown_until = time.monotonic() + self.cooldown_seconds
            self._primary_error = reason
        if should_log:
            logger.error(
                "Primary model %s authentication failed; using fallback for %.0f seconds: %s",
                self.primary_model_id,
                self.cooldown_seconds,
                reason,
            )
        self._mark_fallback(reason)

    def _mark_fallback(self, reason: str) -> None:
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree:
                run_tree.metadata["served_by_fallback"] = True
                run_tree.metadata["primary_model_error"] = reason
                run_tree.tags = list(run_tree.tags or [])
                if "served_by_fallback" not in run_tree.tags:
                    run_tree.tags.append("served_by_fallback")
        except Exception:
            logger.debug("Unable to annotate fallback run metadata", exc_info=True)

    async def _fallback_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        reason: str,
    ) -> ModelCallResult:
        last_exception: Exception | None = None
        self._mark_fallback(reason)
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except Exception as exception:
                last_exception = exception
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Route authentication failures to fallbacks and remember the outage."""
        if self._primary_is_active(request):
            active, reason = self._cooldown_state()
            if active:
                return await self._fallback_call(request, handler, reason)

            async def observing_handler(observed_request: ModelRequest) -> ModelResponse:
                try:
                    return await handler(observed_request)
                except Exception as exception:
                    if observed_request is request and is_authentication_error(exception):
                        self._record_auth_failure(exception)
                    raise

            try:
                return await super().awrap_model_call(request, observing_handler)
            except Exception:
                raise
        return await super().awrap_model_call(request, handler)

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelCallResult:
        """Route synchronous model calls through the authentication-aware fallback."""
        if self._primary_is_active(request):
            active, reason = self._cooldown_state()
            if active:
                self._mark_fallback(reason)
                return super().wrap_model_call(
                    request.override(model=self.models[0]), handler
                )

            def observing_handler(observed_request: ModelRequest):
                try:
                    return handler(observed_request)
                except Exception as exception:
                    if observed_request is request and is_authentication_error(exception):
                        self._record_auth_failure(exception)
                    raise

            return super().wrap_model_call(request, observing_handler)
        return super().wrap_model_call(request, handler)


__all__ = ["AuthenticationAwareModelFallbackMiddleware"]
