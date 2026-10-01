"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import threading
import time
from typing import Awaitable, Callable

import langsmith as ls
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.retry import RunnableRetry
from langgraph.errors import GraphBubbleUp
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


def is_permanent_auth_error(exception: BaseException) -> bool:
    """Return whether an exception represents a permanent provider auth failure."""
    status_code = getattr(exception, "status_code", None)
    if status_code is None:
        response = getattr(exception, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return True

    details = str(exception).upper()
    return "API_KEY_INVALID" in details


class PrimaryModelCircuitBreaker:
    """Track permanent primary-model auth failures for one process."""

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        """Configure the cooldown after a permanent auth failure."""
        self.cooldown_seconds = cooldown_seconds
        self._opened_until = 0.0
        self._logged = False
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        """Return whether primary calls are currently blocked."""
        with self._lock:
            return time.monotonic() < self._opened_until

    def record_auth_failure(self, exception: BaseException) -> None:
        """Open the breaker and log the first permanent auth failure."""
        with self._lock:
            self._opened_until = time.monotonic() + self.cooldown_seconds
            if not self._logged:
                logger.error("Primary model unavailable due to authentication failure: %s", exception)
                self._logged = True

    def reset(self) -> None:
        """Reset breaker state for tests and controlled recovery."""
        with self._lock:
            self._opened_until = 0.0
            self._logged = False


_PRIMARY_MODEL_BREAKER = PrimaryModelCircuitBreaker()


def _mark_auth_fallback() -> None:
    try:
        run_tree = ls.get_current_run_tree()
        if run_tree:
            run_tree.metadata["primary_model_unavailable"] = True
            run_tree.metadata["fallback_reason"] = "auth_error"
            if "primary_model_unavailable" not in run_tree.tags:
                run_tree.tags.append("primary_model_unavailable")
    except Exception:
        pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not isinstance(exception, ValueError)
            and not is_permanent_auth_error(exception)
        )
        return kwargs


class AuthAwareRunnableFallback(Runnable):
    """Use fallbacks directly while the primary auth circuit is open."""

    def __init__(
        self,
        primary: Runnable,
        fallbacks: list[Runnable],
        breaker: PrimaryModelCircuitBreaker | None = None,
    ) -> None:
        """Configure a primary runnable and its fallback chain."""
        self.primary = primary
        self.runnable = primary
        self.fallbacks = fallbacks
        self.breaker = breaker or _PRIMARY_MODEL_BREAKER

    def _invoke_fallbacks(self, input: object, config: RunnableConfig | None, **kwargs: object) -> object:
        last_exception: Exception | None = None
        for fallback in self.fallbacks:
            try:
                return fallback.invoke(input, config=config, **kwargs)
            except Exception as exception:
                last_exception = exception
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    async def _ainvoke_fallbacks(
        self, input: object, config: RunnableConfig | None, **kwargs: object
    ) -> object:
        last_exception: Exception | None = None
        for fallback in self.fallbacks:
            try:
                return await fallback.ainvoke(input, config=config, **kwargs)
            except Exception as exception:
                last_exception = exception
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    def invoke(self, input: object, config: RunnableConfig | None = None, **kwargs: object) -> object:
        """Invoke the primary model or the configured fallback chain."""
        if self.breaker.is_open():
            _mark_auth_fallback()
            return self._invoke_fallbacks(input, config, **kwargs)
        try:
            return self.primary.invoke(input, config=config, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exception:
            if is_permanent_auth_error(exception):
                self.breaker.record_auth_failure(exception)
                _mark_auth_fallback()
            return self._invoke_fallbacks(input, config, **kwargs)

    async def ainvoke(
        self, input: object, config: RunnableConfig | None = None, **kwargs: object
    ) -> object:
        """Invoke the primary model or the configured fallback chain asynchronously."""
        if self.breaker.is_open():
            _mark_auth_fallback()
            return await self._ainvoke_fallbacks(input, config, **kwargs)
        try:
            return await self.primary.ainvoke(input, config=config, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exception:
            if is_permanent_auth_error(exception):
                self.breaker.record_auth_failure(exception)
                _mark_auth_fallback()
            return await self._ainvoke_fallbacks(input, config, **kwargs)


class AuthAwareModelFallbackMiddleware(AgentMiddleware):
    """Fallback middleware that opens a circuit for permanent auth failures."""

    def __init__(
        self,
        *models: object,
        breaker: PrimaryModelCircuitBreaker | None = None,
    ) -> None:
        """Configure fallback models and the process-level breaker."""
        from langchain.agents.middleware import ModelFallbackMiddleware

        self._middleware = ModelFallbackMiddleware(*models)
        self.models = self._middleware.models
        self.breaker = breaker or _PRIMARY_MODEL_BREAKER
        super().__init__()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Route around an unavailable primary model before selecting fallbacks."""
        from langchain.agents.middleware.model_fallback import (
            _sanitize_request_for_fallback,
        )

        auth_failure = self.breaker.is_open()
        last_exception: Exception | None = None
        if not auth_failure:
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exception:
                if is_permanent_auth_error(exception):
                    self.breaker.record_auth_failure(exception)
                    auth_failure = True
                else:
                    last_exception = exception
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                result = handler(fallback_request.override(model=fallback_model))
                if auth_failure:
                    _mark_auth_fallback()
                return result
            except GraphBubbleUp:
                raise
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
        """Route around an unavailable primary model before selecting fallbacks."""
        from langchain.agents.middleware.model_fallback import (
            _sanitize_request_for_fallback,
        )

        auth_failure = self.breaker.is_open()
        last_exception: Exception | None = None
        if not auth_failure:
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exception:
                if is_permanent_auth_error(exception):
                    self.breaker.record_auth_failure(exception)
                    auth_failure = True
                else:
                    last_exception = exception
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                result = await handler(fallback_request.override(model=fallback_model))
                if auth_failure:
                    _mark_auth_fallback()
                return result
            except GraphBubbleUp:
                raise
            except Exception as exception:
                last_exception = exception
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")


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
                if is_permanent_auth_error(e):
                    _PRIMARY_MODEL_BREAKER.record_auth_failure(e)
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
