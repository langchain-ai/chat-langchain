"""Shared authentication-aware model fallback behavior."""

import logging
import threading
import time
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import (
    GraphBubbleUp,
    _sanitize_request_for_fallback,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.runnables import Runnable, RunnableConfig, RunnableWithFallbacks
from pydantic import PrivateAttr

from src.middleware.retry_middleware import _ProviderValidationAwareRunnableRetry

logger = logging.getLogger(__name__)


class ModelAvailabilityState:
    """Track a model unavailable after an authentication failure."""

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        """Configure the authentication cooldown duration."""
        self.cooldown_seconds = cooldown_seconds
        self._unavailable_until = 0.0
        self._lock = threading.Lock()

    def is_unavailable(self) -> bool:
        """Return whether the model is still in its authentication cooldown."""
        with self._lock:
            return time.monotonic() < self._unavailable_until

    def mark_unavailable(self) -> bool:
        """Mark the model unavailable and report whether this starts a cooldown."""
        with self._lock:
            now = time.monotonic()
            if now < self._unavailable_until:
                return False
            self._unavailable_until = now + self.cooldown_seconds
            return True


def is_authentication_failure(error: BaseException) -> bool:
    """Identify provider errors that indicate invalid or unauthorized credentials."""
    text = str(error).upper()
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return (
        status_code in {401, 403}
        or response_status in {401, 403}
        or any(
            marker in text
            for marker in (
                "API_KEY_INVALID",
                "INVALID API KEY",
                "INVALID_API_KEY",
                "UNAUTHENTICATED",
                "UNAUTHORIZED",
                "PERMISSION_DENIED",
                "AUTHENTICATION",
                "401",
                "403",
            )
        )
    )


class AuthenticationAwareRetry(_ProviderValidationAwareRunnableRetry):
    """Mark a shared model state when retry exhaustion reveals auth failure."""

    _availability: ModelAvailabilityState = PrivateAttr()
    _model_id: str = PrivateAttr()

    def __init__(
        self,
        *args: Any,
        availability: ModelAvailabilityState,
        model_id: str,
        **kwargs: Any,
    ) -> None:
        """Configure retry state tracking for one model."""
        super().__init__(*args, **kwargs)
        self._availability = availability
        self._model_id = model_id

    def _record_auth_failure(self, error: BaseException) -> None:
        if is_authentication_failure(error) and self._availability.mark_unavailable():
            logger.warning(
                "Model %s unavailable due to authentication failure; using fallback for %.0fs",
                self._model_id,
                self._availability.cooldown_seconds,
            )

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        """Invoke the retrying model and record authentication failures."""
        try:
            return super().invoke(input, config, **kwargs)
        except Exception as error:
            self._record_auth_failure(error)
            raise

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the retrying model asynchronously and record auth failures."""
        try:
            return await super().ainvoke(input, config, **kwargs)
        except Exception as error:
            self._record_auth_failure(error)
            raise


class AuthenticationAwareRunnableWithFallbacks(RunnableWithFallbacks):
    """Skip an unavailable primary before using the configured fallback chain."""

    _availability: ModelAvailabilityState = PrivateAttr()

    def __init__(
        self,
        primary: Runnable,
        fallbacks: list[Runnable],
        availability: ModelAvailabilityState,
    ) -> None:
        """Configure a primary runnable, its fallbacks, and shared state."""
        super().__init__(runnable=primary, fallbacks=fallbacks)
        self._availability = availability

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        """Use the first fallback directly while the primary is unavailable."""
        availability = self.__pydantic_private__["_availability"]
        if availability.is_unavailable():
            return self.fallbacks[0].invoke(input, config, **kwargs)
        return super().invoke(input, config, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Use the first fallback directly while the primary is unavailable."""
        availability = self.__pydantic_private__["_availability"]
        if availability.is_unavailable():
            return await self.fallbacks[0].ainvoke(input, config, **kwargs)
        return await super().ainvoke(input, config, **kwargs)


class AuthenticationAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Skip the primary model after authentication failure during a cooldown."""

    def __init__(
        self,
        primary_model_id: str,
        availability: ModelAvailabilityState,
        first_model: str,
        *additional_models: str,
    ) -> None:
        """Configure fallback models and shared primary availability state."""
        super().__init__(first_model, *additional_models)
        self.primary_model_id = primary_model_id
        self.availability = availability

    def _fallback_request(self, request: ModelRequest, model: Any) -> ModelRequest:
        return _sanitize_request_for_fallback(request, model).override(model=model)

    def _mark_auth_failure(self, error: BaseException) -> None:
        if is_authentication_failure(error) and self.availability.mark_unavailable():
            logger.warning(
                "Model %s unavailable due to authentication failure; using fallback for %.0fs",
                self.primary_model_id,
                self.availability.cooldown_seconds,
            )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Route calls around a shared primary authentication cooldown."""
        if self.availability.is_unavailable():
            return handler(self._fallback_request(request, self.models[0]))
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            self._mark_auth_failure(error)
            return self._run_fallbacks(request, handler, error)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Route async calls around a shared primary authentication cooldown."""
        if self.availability.is_unavailable():
            return await handler(self._fallback_request(request, self.models[0]))
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            self._mark_auth_failure(error)
            return await self._arun_fallbacks(request, handler, error)

    def _run_fallbacks(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        first_error: Exception,
    ) -> ModelResponse:
        last_error = first_error
        for model in self.models:
            try:
                return handler(self._fallback_request(request, model))
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_error = error
        raise last_error

    async def _arun_fallbacks(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        first_error: Exception,
    ) -> ModelResponse:
        last_error = first_error
        for model in self.models:
            try:
                return await handler(self._fallback_request(request, model))
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_error = error
        raise last_error


__all__ = [
    "AuthenticationAwareModelFallbackMiddleware",
    "AuthenticationAwareRetry",
    "AuthenticationAwareRunnableWithFallbacks",
    "ModelAvailabilityState",
    "is_authentication_failure",
]
