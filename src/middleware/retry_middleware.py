"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import _sanitize_request_for_fallback
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables.retry import RunnableRetry
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


def is_permanent_auth_failure(exception: BaseException) -> bool:
    """Return whether an exception indicates rejected provider credentials."""
    status_codes = {
        getattr(exception, "status_code", None),
        getattr(exception, "code", None),
        getattr(getattr(exception, "response", None), "status_code", None),
    }
    if status_codes & {401, 403}:
        return True
    text = str(exception).upper()
    return "API_KEY_INVALID" in text or "API KEY NOT VALID" in text


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not isinstance(exception, ValueError)
            and not is_permanent_auth_failure(exception)
        )
        return kwargs


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
                if isinstance(e, ValueError) or is_permanent_auth_failure(e):
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


class ModelAuthFallbackMiddleware(ModelFallbackMiddleware):
    """Route model calls around a process-level authentication circuit breaker."""

    _circuit_opened_at: float | None = None
    _circuit_lock = threading.Lock()

    def __init__(
        self,
        primary_model_id: str,
        *fallback_models: str,
        cooldown_seconds: float = 300.0,
    ) -> None:
        """Configure the fallback chain and authentication cooldown."""
        super().__init__(*fallback_models)
        self.primary_model_id = primary_model_id
        self.cooldown_seconds = cooldown_seconds
        self.fallback_model_ids = fallback_models

    @classmethod
    def reset_circuit(cls) -> None:
        """Reset the process-level authentication circuit."""
        with cls._circuit_lock:
            cls._circuit_opened_at = None

    @classmethod
    def _circuit_is_open(cls, cooldown_seconds: float) -> bool:
        with cls._circuit_lock:
            if cls._circuit_opened_at is None:
                return False
            if time.monotonic() - cls._circuit_opened_at >= cooldown_seconds:
                cls._circuit_opened_at = None
                return False
            return True

    @classmethod
    def _open_circuit(cls, exception: BaseException) -> None:
        with cls._circuit_lock:
            if cls._circuit_opened_at is None:
                cls._circuit_opened_at = time.monotonic()
                logger.error("Primary model authentication failed; opening circuit: %s", exception)

    @staticmethod
    def _record_served_model(model_id: str, fallback_reason: str | None = None) -> None:
        try:
            from langgraph.config import get_config

            metadata = get_config().setdefault("metadata", {})
            metadata["served_by_model"] = model_id
            if fallback_reason:
                metadata["fallback_reason"] = fallback_reason
            else:
                metadata.pop("fallback_reason", None)
        except RuntimeError:
            return

    def _fallback_request(
        self,
        request: ModelRequest,
        fallback_model: object,
        model_id: str,
        reason: str,
    ) -> ModelRequest:
        request = _sanitize_request_for_fallback(request, fallback_model)
        self._record_served_model(model_id, reason)
        return request.override(model=fallback_model)

    def _primary_request(self, request: ModelRequest) -> ModelRequest:
        self._record_served_model(self.primary_model_id)
        return request

    def wrap_model_call(self, request, handler):
        """Call the primary model or route around an open auth circuit."""
        if self._circuit_is_open(self.cooldown_seconds):
            return handler(
                self._fallback_request(
                    request,
                    self.models[0],
                    self.fallback_model_ids[0],
                    "primary_auth_circuit_open",
                )
            )
        try:
            response = handler(self._primary_request(request))
            self.reset_circuit()
            return response
        except Exception as exception:
            if is_permanent_auth_failure(exception):
                self._open_circuit(exception)
                reason = "primary_auth_failure"
            else:
                reason = "primary_failure"
            last_exception = exception
            for fallback_model, model_id in zip(self.models, self.fallback_model_ids):
                try:
                    return handler(self._fallback_request(request, fallback_model, model_id, reason))
                except Exception as fallback_exception:
                    last_exception = fallback_exception
            raise last_exception

    async def awrap_model_call(self, request, handler):
        """Async variant of the authentication-aware fallback boundary."""
        if self._circuit_is_open(self.cooldown_seconds):
            return await handler(
                self._fallback_request(
                    request,
                    self.models[0],
                    self.fallback_model_ids[0],
                    "primary_auth_circuit_open",
                )
            )
        try:
            response = await handler(self._primary_request(request))
            self.reset_circuit()
            return response
        except Exception as exception:
            if is_permanent_auth_failure(exception):
                self._open_circuit(exception)
                reason = "primary_auth_failure"
            else:
                reason = "primary_failure"
            last_exception = exception
            for fallback_model, model_id in zip(self.models, self.fallback_model_ids):
                try:
                    return await handler(
                        self._fallback_request(request, fallback_model, model_id, reason)
                    )
                except Exception as fallback_exception:
                    last_exception = fallback_exception
            raise last_exception


__all__ = [
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "ModelAuthFallbackMiddleware",
    "is_permanent_auth_failure",
]
