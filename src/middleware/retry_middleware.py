"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import os
import threading
import time
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain_core.runnables.retry import RunnableRetry
from pydantic import PrivateAttr
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

AUTH_COOLDOWN_SECONDS = float(os.getenv("MODEL_AUTH_COOLDOWN_SECONDS", "300"))
_provider_breakers: dict[str, float] = {}
_provider_breakers_lock = threading.Lock()

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class ProviderCircuitOpenError(Exception):
    """Raised when a provider is cooling down after an authentication failure."""


def is_auth_error(exception: BaseException) -> bool:
    """Return whether an exception represents rejected provider credentials."""
    class_name = exception.__class__.__name__.lower()
    status_code = getattr(exception, "status_code", None)
    response = getattr(exception, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code in (401, 403) or "authenticationerror" in class_name:
        return True
    if "permissiondeniederror" in class_name or "unauthenticated" in class_name:
        return True
    text = str(exception).upper()
    return any(
        marker in text
        for marker in (
            "API_KEY_INVALID",
            "PERMISSION_DENIED",
            "UNAUTHENTICATED",
            "API KEY NOT VALID",
        )
    )


def provider_is_available(provider: str) -> bool:
    """Return whether the provider is outside its authentication cooldown."""
    with _provider_breakers_lock:
        tripped_at = _provider_breakers.get(provider)
        if tripped_at is None:
            return True
        if time.monotonic() - tripped_at >= AUTH_COOLDOWN_SECONDS:
            del _provider_breakers[provider]
            return True
        return False


def trip_provider(provider: str) -> None:
    """Trip a provider circuit after its first authentication failure."""
    with _provider_breakers_lock:
        if provider not in _provider_breakers:
            _provider_breakers[provider] = time.monotonic()
            logger.warning("Provider %s disabled after an authentication failure", provider)


def reset_provider_breakers() -> None:
    """Clear provider circuits for tests and process reconfiguration."""
    with _provider_breakers_lock:
        _provider_breakers.clear()


def model_provider(model: object) -> str:
    """Identify a model provider from its LangChain model metadata."""
    model_id = getattr(model, "model_name", None) or getattr(model, "model", None)
    llm_type = getattr(model, "_llm_type", None) or ""
    text = f"{model_id or ''}:{llm_type}".lower()
    for provider in ("google", "openai", "anthropic", "baseten"):
        if provider in text:
            return provider
    return llm_type or str(model_id or "unknown")


def _record_served_model(model: object) -> None:
    """Record the successful model in the active run metadata."""
    try:
        from langgraph.config import get_config

        config = get_config()
        config.setdefault("metadata", {})["served_model"] = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or model_provider(model)
        )
    except RuntimeError:
        return


class AuthAwareModelFallbackMiddleware(AgentMiddleware):
    """Skip tripped providers and mark the model that serves each turn."""

    def __init__(self, *model_ids: str):
        """Initialize fallback model instances."""
        super().__init__()
        self.models = [init_chat_model(model_id) for model_id in model_ids]

    def _call(self, request, handler):
        models = (request.model, *self.models)
        last_exception = None
        for model in models:
            provider = model_provider(model)
            if not provider_is_available(provider):
                continue
            try:
                response = handler(request.override(model=model))
                _record_served_model(model)
                return response
            except Exception as exception:
                last_exception = exception
                if is_auth_error(exception):
                    trip_provider(provider)
        if last_exception:
            raise last_exception
        raise ProviderCircuitOpenError(model_provider(request.model))

    async def _acall(self, request, handler):
        models = (request.model, *self.models)
        last_exception = None
        for model in models:
            provider = model_provider(model)
            if not provider_is_available(provider):
                continue
            try:
                response = await handler(request.override(model=model))
                _record_served_model(model)
                return response
            except Exception as exception:
                last_exception = exception
                if is_auth_error(exception):
                    trip_provider(provider)
        if last_exception:
            raise last_exception
        raise ProviderCircuitOpenError(model_provider(request.model))

    def wrap_model_call(self, request, handler):
        """Call the first available model."""
        return self._call(request, handler)

    async def awrap_model_call(self, request, handler):
        """Asynchronously call the first available model."""
        return await self._acall(request, handler)


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    _provider: str | None = PrivateAttr(default=None)

    def __init__(self, *, provider: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._provider = provider

    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not isinstance(exception, ValueError)
            and not is_auth_error(exception)
        )
        return kwargs

    def invoke(self, input, config=None, **kwargs):
        if self._provider and not provider_is_available(self._provider):
            raise ProviderCircuitOpenError(self._provider)
        try:
            return super().invoke(input, config, **kwargs)
        except Exception as exception:
            if self._provider and is_auth_error(exception):
                trip_provider(self._provider)
            raise

    async def ainvoke(self, input, config=None, **kwargs):
        if self._provider and not provider_is_available(self._provider):
            raise ProviderCircuitOpenError(self._provider)
        try:
            return await super().ainvoke(input, config, **kwargs)
        except Exception as exception:
            if self._provider and is_auth_error(exception):
                trip_provider(self._provider)
            raise


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
                if isinstance(e, ValueError) or is_auth_error(e):
                    if is_auth_error(e):
                        trip_provider(model_provider(request.model))
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


__all__ = [
    "AUTH_COOLDOWN_SECONDS",
    "AuthAwareModelFallbackMiddleware",
    "MalformedResponseError",
    "ModelRetryMiddleware",
    "ProviderCircuitOpenError",
    "_ProviderValidationAwareRunnableRetry",
    "is_auth_error",
    "model_provider",
    "provider_is_available",
    "reset_provider_breakers",
    "trip_provider",
]
