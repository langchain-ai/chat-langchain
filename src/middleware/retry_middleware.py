"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import os
import threading
import time
from contextvars import ContextVar
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables.retry import RunnableRetry
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""


class PrimaryUnhealthyError(Exception):
    """Raised before invoking a primary model in its cooldown window."""


class _PrimaryCircuitBreaker:
    def __init__(self) -> None:
        self.cooldown = float(os.getenv("PRIMARY_MODEL_COOLDOWN_SECONDS", "60"))
        self._unhealthy_until: dict[str, float] = {}
        self._errors: dict[str, Exception] = {}
        self._lock = threading.Lock()

    def check(self, model_id: str) -> None:
        with self._lock:
            until = self._unhealthy_until.get(model_id, 0)
            if until <= time.monotonic():
                self._unhealthy_until.pop(model_id, None)
                self._errors.pop(model_id, None)
                return
            error = self._errors.get(model_id)
        if error is not None:
            raise error
        raise PrimaryUnhealthyError(f"Primary model {model_id} is unhealthy")

    def record_failure(self, model_id: str, error: Exception) -> None:
        with self._lock:
            self._unhealthy_until[model_id] = time.monotonic() + self.cooldown
            self._errors[model_id] = error

    def record_success(self, model_id: str) -> None:
        with self._lock:
            self._unhealthy_until.pop(model_id, None)
            self._errors.pop(model_id, None)

    def is_unhealthy(self, model_id: str) -> bool:
        try:
            self.check(model_id)
        except Exception:
            return True
        return False


primary_circuit_breaker = _PrimaryCircuitBreaker()
_logged_auth_failures: set[str] = set()
_logged_auth_failures_lock = threading.Lock()


def _nested_error_values(value: object, seen: set[int] | None = None) -> list[object]:
    if seen is None:
        seen = set()
    if id(value) in seen:
        return []
    seen.add(id(value))
    values = [value]
    if isinstance(value, BaseException):
        values.extend(value.args)
        values.extend(
            nested
            for nested in (
                getattr(value, "__cause__", None),
                getattr(value, "__context__", None),
                getattr(value, "response", None),
                getattr(value, "body", None),
                getattr(value, "details", None),
            )
            if nested is not None
        )
    elif isinstance(value, dict):
        values.extend(value.keys())
        values.extend(value.values())
    elif isinstance(value, (list, tuple, set)):
        values.extend(value)
    return values


def _is_auth_error(exc: BaseException) -> bool:
    values = _nested_error_values(exc)
    text = " ".join(str(value) for value in values).lower()
    if "api_key_invalid" in text or "api key not valid" in text:
        return True
    status_codes = {
        getattr(value, attribute, None)
        for value in values
        for attribute in ("status_code", "status", "http_status")
    }
    if status_codes & {401, 403, "401", "403"}:
        return True
    auth_terms = ("authentication", "permission denied", "unauthorized", "forbidden")
    auth_types = ("authenticationerror", "permissionerror", "unauthorized", "forbidden")
    return any(term in text for term in auth_terms) or any(
        term in type(value).__name__.lower() for value in values for term in auth_types
    )


def log_auth_failure_once(model_id: str) -> None:
    """Log one credential failure for a model per process."""
    with _logged_auth_failures_lock:
        if model_id in _logged_auth_failures:
            return
        _logged_auth_failures.add(model_id)
    logger.error("Credential authentication failed for model %s", model_id)


def record_served_model(model_id: str, served_by_fallback: bool) -> None:
    """Record the model that produced the current answer."""
    _served_model_state.set((model_id, served_by_fallback))


def get_served_model_metadata() -> dict[str, object]:
    """Return metadata for the model that produced the current answer."""
    model = _served_model_state.get()
    if model is None:
        return {}
    return {"served_model": model[0], "served_by_fallback": model[1]}


_served_model_state: ContextVar[tuple[str, bool] | None] = ContextVar(
    "served_model_state", default=None
)


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    def __init__(self, *args: object, model_id: str | None = None, **kwargs: object):
        super().__init__(*args, **kwargs)
        self._model_id = model_id

    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: (
                not isinstance(exception, ValueError) and not _is_auth_error(exception)
            )
        )
        return kwargs

    def invoke(self, input: object, config: object = None, **kwargs: object) -> object:
        if self._model_id and self._model_id in _PRIMARY_MODEL_IDS:
            primary_circuit_breaker.check(self._model_id)
        try:
            result = super().invoke(input, config, **kwargs)
        except Exception as exc:
            if _is_auth_error(exc):
                if self._model_id in _PRIMARY_MODEL_IDS:
                    primary_circuit_breaker.record_failure(self._model_id, exc)
                log_auth_failure_once(self._model_id or "unknown")
            raise
        if self._model_id:
            if self._model_id in _PRIMARY_MODEL_IDS:
                primary_circuit_breaker.record_success(self._model_id)
            record_served_model(
                self._model_id, self._model_id not in _PRIMARY_MODEL_IDS
            )
        return result

    async def ainvoke(
        self, input: object, config: object = None, **kwargs: object
    ) -> object:
        if self._model_id and self._model_id in _PRIMARY_MODEL_IDS:
            primary_circuit_breaker.check(self._model_id)
        try:
            result = await super().ainvoke(input, config, **kwargs)
        except Exception as exc:
            if _is_auth_error(exc):
                if self._model_id in _PRIMARY_MODEL_IDS:
                    primary_circuit_breaker.record_failure(self._model_id, exc)
                log_auth_failure_once(self._model_id or "unknown")
            raise
        if self._model_id:
            if self._model_id in _PRIMARY_MODEL_IDS:
                primary_circuit_breaker.record_success(self._model_id)
            record_served_model(
                self._model_id, self._model_id not in _PRIMARY_MODEL_IDS
            )
        return result


_PRIMARY_MODEL_IDS: set[str] = set()


class PrimaryAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Skip an unhealthy primary before trying configured fallback models."""

    def __init__(self, primary_model_id: str, *fallback_models: str) -> None:
        """Initialize fallback models and the shared primary circuit breaker."""
        if fallback_models:
            super().__init__(*fallback_models)
        else:
            AgentMiddleware.__init__(self)
            self.models = []
        self.primary_model_id = primary_model_id
        _PRIMARY_MODEL_IDS.add(primary_model_id)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable
    ) -> ModelResponse:
        """Skip an unhealthy primary and serve the first working fallback."""
        last_exception: Exception | None = None
        try:
            primary_circuit_breaker.check(self.primary_model_id)
            result = handler(request)
        except Exception as exc:
            last_exception = exc
            if _is_auth_error(exc):
                primary_circuit_breaker.record_failure(self.primary_model_id, exc)
                log_auth_failure_once(self.primary_model_id)
        else:
            primary_circuit_breaker.record_success(self.primary_model_id)
            record_served_model(self.primary_model_id, False)
            return result
        return self._run_fallbacks(request, handler, last_exception)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable
    ) -> ModelResponse:
        """Asynchronously skip an unhealthy primary and use a fallback."""
        last_exception: Exception | None = None
        try:
            primary_circuit_breaker.check(self.primary_model_id)
            result = await handler(request)
        except Exception as exc:
            last_exception = exc
            if _is_auth_error(exc):
                primary_circuit_breaker.record_failure(self.primary_model_id, exc)
                log_auth_failure_once(self.primary_model_id)
        else:
            primary_circuit_breaker.record_success(self.primary_model_id)
            record_served_model(self.primary_model_id, False)
            return result
        return await self._run_fallbacks_async(request, handler, last_exception)

    def _run_fallbacks(
        self, request: ModelRequest, handler: Callable, last_exception: Exception | None
    ):
        if last_exception is None:
            last_exception = PrimaryUnhealthyError(self.primary_model_id)
        for fallback_model in self.models:
            try:
                result = handler(request.override(model=fallback_model))
            except Exception as exc:
                last_exception = exc
                continue
            record_served_model(_model_name(fallback_model), True)
            return result
        raise last_exception

    async def _run_fallbacks_async(
        self, request: ModelRequest, handler: Callable, last_exception: Exception | None
    ):
        if last_exception is None:
            last_exception = PrimaryUnhealthyError(self.primary_model_id)
        for fallback_model in self.models:
            try:
                result = await handler(request.override(model=fallback_model))
            except Exception as exc:
                last_exception = exc
                continue
            record_served_model(_model_name(fallback_model), True)
            return result
        raise last_exception


def _model_name(model: object) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "name", None)
        or type(model).__name__
    )


def _matches_model_id(model: object, model_id: str) -> bool:
    model_name = _model_name(model)
    return model_name == model_id or model_name == model_id.rsplit(":", 1)[-1]


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient model failures and malformed responses."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        primary_model_id: str | None = None,
    ):
        """Configure retry attempts and backoff timing."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.primary_model_id = primary_model_id

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        metadata = getattr(response, "response_metadata", None) or {}
        return metadata.get("finish_reason", "")

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable
    ) -> ModelCallResult:
        """Retry transient synchronous model failures."""
        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = handler(request)
                finish_reason = self._get_finish_reason(response)
                if (
                    finish_reason in RETRYABLE_FINISH_REASONS
                    and attempt < self.max_retries
                ):
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        "Retryable response (%s), retrying in %.2fs",
                        finish_reason,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if self.primary_model_id and _matches_model_id(
                    request.model, self.primary_model_id
                ):
                    record_served_model(self.primary_model_id, False)
                return response
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                if _is_auth_error(exc):
                    model_id = self.primary_model_id or _model_name(request.model)
                    if model_id == self.primary_model_id and _matches_model_id(
                        request.model, self.primary_model_id
                    ):
                        primary_circuit_breaker.record_failure(model_id, exc)
                    log_auth_failure_once(model_id)
                    raise
                last_exception = exc
                if attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning("Model call failed, retrying in %.2fs", delay)
                    time.sleep(delay)
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected state in retry middleware")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry transient asynchronous model failures."""
        last_exception: Exception | None = None
        last_retryable_reason: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)
                if (
                    finish_reason in RETRYABLE_FINISH_REASONS
                    and attempt < self.max_retries
                ):
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        "Retryable response (%s) attempt %s/%s, retrying in %.2fs",
                        finish_reason,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    last_retryable_reason = finish_reason
                    await asyncio.sleep(delay)
                    continue
                if self.primary_model_id and _matches_model_id(
                    request.model, self.primary_model_id
                ):
                    record_served_model(self.primary_model_id, False)
                return response
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                if _is_auth_error(exc):
                    model_id = self.primary_model_id or _model_name(request.model)
                    if model_id == self.primary_model_id and _matches_model_id(
                        request.model, self.primary_model_id
                    ):
                        primary_circuit_breaker.record_failure(model_id, exc)
                    log_auth_failure_once(model_id)
                    raise
                last_exception = exc
                if attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        "Model call failed attempt %s/%s, retrying in %.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Model call failed after %s attempts: %s",
                        self.max_retries + 1,
                        exc,
                    )
        if last_exception:
            raise last_exception
        if last_retryable_reason:
            raise MalformedResponseError(
                f"Model returned {last_retryable_reason} after {self.max_retries + 1} attempts"
            )
        raise RuntimeError("Unexpected state in retry middleware")


__all__ = [
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "PrimaryAwareModelFallbackMiddleware",
    "PrimaryUnhealthyError",
    "_ProviderValidationAwareRunnableRetry",
    "_is_auth_error",
    "get_served_model_metadata",
    "log_auth_failure_once",
    "primary_circuit_breaker",
    "record_served_model",
]
