"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import time
from threading import Lock
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables import Runnable
from langchain_core.runnables.retry import RunnableRetry
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


def is_authentication_error(exception: BaseException) -> bool:
    """Return whether an exception indicates provider authentication failure."""
    status_code = getattr(exception, "status_code", None)
    response = getattr(exception, "response", None)
    response_status = getattr(response, "status_code", None)
    if status_code in {401, 403} or response_status in {401, 403}:
        return True

    exception_name = type(exception).__name__.lower()
    if any(
        marker in exception_name
        for marker in ("authentication", "permissiondenied", "unauthorized", "forbidden")
    ):
        return True

    details = repr(exception).upper()
    return any(
        marker in details
        for marker in (
            "API_KEY_INVALID",
            "INVALID_API_KEY",
            "PERMISSION_DENIED",
            "UNAUTHENTICATED",
            "UNAUTHORIZED",
            "HTTP 401",
            "HTTP 403",
        )
    )


class PrimaryModelCircuitOpen(Exception):
    """Raised when the primary model authentication circuit is open."""


class PrimaryModelAuthCircuit:
    """Circuit breaker for authentication failures from one primary model."""

    def __init__(self, provider: str, model: str, cooldown_seconds: float = 300):
        self.provider = provider
        self.model = model
        self.cooldown_seconds = cooldown_seconds
        self._open_until = 0.0
        self._lock = Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until

    def allow_request(self) -> bool:
        return not self.is_open

    def trip(self, exception: BaseException) -> None:
        with self._lock:
            now = time.monotonic()
            already_open = now < self._open_until
            self._open_until = now + self.cooldown_seconds
        if not already_open:
            logger.error(
                "Primary model authentication failed; bypassing %s:%s for %.0f seconds: %s",
                self.provider,
                self.model,
                self.cooldown_seconds,
                exception,
            )


class AuthenticationAwareRunnable(Runnable):
    """Wrap a primary runnable with authentication circuit-breaker behavior."""

    def __init__(self, bound: Runnable, circuit: PrimaryModelAuthCircuit):
        self.bound = bound
        self.circuit = circuit
        self.max_attempt_number = getattr(bound, "max_attempt_number", None)

    def invoke(self, input: object, config: object | None = None, **kwargs: object) -> object:
        if not self.circuit.allow_request():
            raise PrimaryModelCircuitOpen(self.circuit.model)
        try:
            return self.bound.invoke(input, config=config, **kwargs)
        except Exception as exception:
            if is_authentication_error(exception):
                self.circuit.trip(exception)
            raise

    async def ainvoke(
        self, input: object, config: object | None = None, **kwargs: object
    ) -> object:
        if not self.circuit.allow_request():
            raise PrimaryModelCircuitOpen(self.circuit.model)
        try:
            return await self.bound.ainvoke(input, config=config, **kwargs)
        except Exception as exception:
            if is_authentication_error(exception):
                self.circuit.trip(exception)
            raise


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: (
                not isinstance(exception, ValueError)
                and not is_authentication_error(exception)
            )
        )
        return kwargs


class AuthenticationAwareModelFallbackMiddleware(AgentMiddleware):
    """Add authentication circuit-breaker behavior to model fallbacks."""

    def __init__(
        self,
        primary_model: str,
        circuit: PrimaryModelAuthCircuit,
        first_model: object,
        *additional_models: object,
    ):
        from langchain.agents.middleware import ModelFallbackMiddleware

        self._fallback = ModelFallbackMiddleware(first_model, *additional_models)
        self.models = self._fallback.models
        self.primary_model = primary_model
        self.circuit = circuit
        super().__init__()

    def _is_primary(self, request: ModelRequest) -> bool:
        model = getattr(request, "model", None)
        model_id = getattr(model, "model", None) or getattr(model, "model_name", None)
        return model_id in {self.primary_model, self.primary_model.split(":", 1)[-1]}

    def _call_handler(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        if self._is_primary(request) and not self.circuit.allow_request():
            raise PrimaryModelCircuitOpen(self.primary_model)
        try:
            response = handler(request)
        except Exception as exception:
            if self._is_primary(request) and is_authentication_error(exception):
                self.circuit.trip(exception)
            raise
        record_answered_by_model(request)
        return response

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        return self._fallback.wrap_model_call(request, lambda current: self._call_handler(current, handler))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        return await self._fallback.awrap_model_call(
            request,
            lambda current: self._acall_handler(current, handler),
        )

    async def _acall_handler(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        if self._is_primary(request) and not self.circuit.allow_request():
            raise PrimaryModelCircuitOpen(self.primary_model)
        try:
            response = await handler(request)
        except Exception as exception:
            if self._is_primary(request) and is_authentication_error(exception):
                self.circuit.trip(exception)
            raise
        record_answered_by_model(request)
        return response


def record_answered_by_model(request: ModelRequest) -> None:
    """Record the model that produced a successful answer on the root run."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return
        while run.parent_run is not None:
            run = run.parent_run
        model = getattr(request.model, "model", None) or getattr(
            request.model, "model_name", None
        )
        if model:
            run.metadata["answered_by_model"] = model
    except Exception:
        logger.debug("Unable to record answering model metadata", exc_info=True)


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient model failures and malformed responses."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        """Configure retry attempts and backoff timing."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        metadata = getattr(response, "response_metadata", None) or {}
        return metadata.get("finish_reason", "")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry transient failures from the wrapped model handler."""
        last_exception: Exception | None = None
        last_retryable_reason: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)

                if finish_reason in RETRYABLE_FINISH_REASONS:
                    if attempt < self.max_retries:
                        delay = self.initial_delay * (self.backoff_factor**attempt)
                        logger.warning(
                            f"Retryable response ({finish_reason}) "
                            f"attempt {attempt + 1}/{self.max_retries + 1}, "
                            f"retrying in {delay:.2f}s"
                        )
                        last_retryable_reason = finish_reason
                        await asyncio.sleep(delay)
                        continue

                return response

            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                if is_authentication_error(e):
                    raise
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        f"Model call failed attempt {attempt + 1}/{self.max_retries + 1}: {e}, "
                        f"retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Model call failed after {self.max_retries + 1} attempts: {e}"
                    )

        # Exhausted retries - raise for fallback middleware
        if last_exception:
            raise last_exception

        if last_retryable_reason:
            raise MalformedResponseError(
                f"Model returned {last_retryable_reason} after {self.max_retries + 1} attempts"
            )

        raise RuntimeError("Unexpected state in retry middleware")


__all__ = ["ModelRetryMiddleware", "MalformedResponseError"]
