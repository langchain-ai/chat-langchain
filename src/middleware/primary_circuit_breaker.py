"""Authentication-aware circuit breaking for the primary model."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest
from langchain_core.language_models import BaseChatModel
from langsmith import set_run_metadata

logger = logging.getLogger(__name__)

_AUTH_REASON = "API_KEY_INVALID"
_STATUS_RE = re.compile(r"(?:status(?:_code)?|http)[^0-9]{0,12}(401|403)", re.I)


def _exception_text(exception: BaseException) -> str:
    values = [str(exception)]
    for attribute in ("message", "reason", "body", "response"):
        value = getattr(exception, attribute, None)
        if value is not None:
            values.append(str(value))
    return " ".join(values)


def is_authentication_error(exception: BaseException) -> bool:
    """Return whether an exception indicates invalid provider credentials."""
    text = _exception_text(exception)
    normalized = text.lower().replace("-", "_")
    if (
        _AUTH_REASON.lower() in normalized
        or "api key invalid" in normalized
        or "api key not valid" in normalized
    ):
        return True
    if _STATUS_RE.search(text):
        return True
    status_codes = (
        getattr(exception, "status_code", None),
        getattr(getattr(exception, "response", None), "status_code", None),
    )
    for value in status_codes:
        if value in {401, 403, "401", "403"}:
            return True
    exception_name = type(exception).__name__.lower()
    if exception_name in {
        "authenticationerror",
        "permissiondeniederror",
        "unauthorizederror",
        "forbiddenerror",
    }:
        return True
    return False


def authentication_error_reason(exception: BaseException) -> str:
    """Return the stable provider reason for an authentication error."""
    text = _exception_text(exception)
    if _AUTH_REASON.lower() in text.lower() or "api key invalid" in text.lower():
        return _AUTH_REASON
    status_match = _STATUS_RE.search(text)
    if status_match:
        return f"HTTP_{status_match.group(1)}"
    return type(exception).__name__


def _model_identifiers(model: BaseChatModel) -> set[str]:
    identifiers = {str(model)}
    for attribute in ("model", "model_name", "model_id"):
        value = getattr(model, attribute, None)
        if value:
            identifiers.add(str(value))
    return {identifier.lower() for identifier in identifiers}


class PrimaryModelCircuitBreaker(AgentMiddleware):
    """Bypass the primary model after an authentication failure."""

    def __init__(
        self,
        primary_model_id: str,
        fallback_model: BaseChatModel,
        cooldown_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the primary circuit and its first fallback."""
        super().__init__()
        self.primary_model_id = primary_model_id.lower()
        self.primary_identifiers = {
            self.primary_model_id,
            self.primary_model_id.split(":")[-1],
        }
        self.fallback_model = fallback_model
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self._opened_at: float | None = None
        self._last_error: BaseException | None = None
        self._last_log_at: float | None = None

    @property
    def is_open(self) -> bool:
        """Return whether the primary circuit is currently open."""
        if self._opened_at is None:
            return False
        if self.clock() - self._opened_at >= self.cooldown_seconds:
            self._opened_at = None
            self._last_error = None
            return False
        return True

    def _is_primary(self, request: ModelRequest) -> bool:
        identifiers = _model_identifiers(request.model)
        return bool(identifiers & self.primary_identifiers)

    def _record_fallback(self, exception: BaseException) -> None:
        set_run_metadata(
            served_by_fallback=True,
            primary_error_class=type(exception).__name__,
            primary_error_reason=authentication_error_reason(exception),
        )

    def open(self, exception: BaseException) -> None:
        """Open the circuit for the configured cooldown period."""
        self._opened_at = self.clock()
        self._last_error = exception
        if self._last_log_at != self._opened_at:
            self._last_log_at = self._opened_at
            logger.error(
                "Primary model circuit opened after authentication failure: %s",
                authentication_error_reason(exception),
            )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        """Guard synchronous primary model calls."""
        if not self._is_primary(request):
            return handler(request)
        if self.is_open:
            exception = self._last_error or RuntimeError("primary circuit is open")
            self._record_fallback(exception)
            return handler(request.override(model=self.fallback_model))
        try:
            return handler(request)
        except Exception as exception:
            if is_authentication_error(exception):
                self.open(exception)
                self._record_fallback(exception)
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Guard asynchronous primary model calls."""
        if not self._is_primary(request):
            return await handler(request)
        if self.is_open:
            exception = self._last_error or RuntimeError("primary circuit is open")
            self._record_fallback(exception)
            return await handler(request.override(model=self.fallback_model))
        try:
            return await handler(request)
        except Exception as exception:
            if is_authentication_error(exception):
                self.open(exception)
                self._record_fallback(exception)
            raise


def startup_probe_enabled() -> bool:
    """Return whether the primary startup probe is enabled."""
    configured = os.getenv("PRIMARY_STARTUP_PROBE")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return any(
        os.getenv(name)
        for name in (
            "LANGGRAPH_DEPLOYMENT_ID",
            "LANGGRAPH_API_URL",
            "LANGSMITH_HOST_REVISION_ID",
            "LANGSMITH_LANGGRAPH_HOST_URL",
        )
    )


def probe_primary_model(model: BaseChatModel, breaker: PrimaryModelCircuitBreaker) -> None:
    """Probe the primary provider and open the breaker on auth rejection."""
    if not startup_probe_enabled():
        return
    try:
        model.invoke("health check")
    except Exception as exception:
        if is_authentication_error(exception):
            breaker.open(exception)
        else:
            logger.warning("Primary provider startup probe failed: %s", exception)
