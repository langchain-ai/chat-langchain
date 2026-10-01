"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import os
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
_logged_authentication_failures: set[tuple[str, str]] = set()

AUTH_VALIDATION_ENV = "MODEL_AUTH_VALIDATION_STRICT"


class ProviderAuthenticationError(BaseException):
    """Raised to prevent authentication failures from entering fallbacks."""


def _exception_values(exception: BaseException) -> list[object]:
    values = [exception, str(exception)]
    for attribute in ("body", "details", "error", "response"):
        value = getattr(exception, attribute, None)
        if value is not None:
            values.append(value)
            if isinstance(value, dict):
                values.extend(value.values())
            else:
                for nested_attribute in ("status_code", "reason", "code", "message"):
                    nested_value = getattr(value, nested_attribute, None)
                    if nested_value is not None:
                        values.append(nested_value)
    return values


def classify_provider_authentication_error(
    exception: BaseException, provider: str | None = None
) -> str | None:
    """Return an authentication failure reason, or None for other errors."""
    values = _exception_values(exception)
    text = " ".join(
        [type(exception).__name__, *(str(value) for value in values)]
    ).upper()
    status_codes = {
        str(getattr(exception, "status_code", "")),
        str(getattr(getattr(exception, "response", None), "status_code", "")),
        str(getattr(exception, "http_status", "")),
    }
    if "401" in status_codes:
        return "HTTP_401_AUTHENTICATION"
    if "403" in status_codes:
        return "HTTP_403_PERMISSION"
    if "API_KEY_INVALID" in text or "API KEY NOT VALID" in text:
        return "API_KEY_INVALID"
    if provider in {"openai", "anthropic"} and any(
        marker in text
        for marker in ("AUTHENTICATIONERROR", "AUTHENTICATION ERROR", "UNAUTHORIZED")
    ):
        return "PROVIDER_AUTHENTICATION_ERROR"
    if any(marker in text for marker in ("INVALID API KEY", "INVALID_API_KEY")):
        return "INVALID_API_KEY"
    return None


def is_provider_authentication_error(
    exception: BaseException, provider: str | None = None
) -> bool:
    """Return whether an exception indicates invalid provider credentials."""
    return classify_provider_authentication_error(exception, provider) is not None


def is_strict_authentication_validation() -> bool:
    """Return whether authentication failures should bypass fallbacks."""
    return os.getenv(AUTH_VALIDATION_ENV, "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class ProviderAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Avoid masking provider authentication failures with fallback models."""

    def _handle_primary_error(
        self, exception: Exception, request: ModelRequest
    ) -> None:
        reason = classify_provider_authentication_error(exception)
        if reason and is_strict_authentication_validation():
            raise ProviderAuthenticationError(reason) from exception
        if reason:
            key = (type(exception).__name__, reason)
            if key not in _logged_authentication_failures:
                _logged_authentication_failures.add(key)
                logger.error(
                    "Falling back after provider authentication failure for %s: %s",
                    getattr(
                        request.model,
                        "model",
                        getattr(request.model, "model_name", "unknown"),
                    ),
                    reason,
                )

    def _fallback_request(
        self, request: ModelRequest, fallback_model: object, reason: str | None
    ) -> ModelRequest:
        if not reason:
            return request
        state = dict(request.state)
        state.update(
            {
                "fallback_service": getattr(
                    fallback_model, "model", str(fallback_model)
                ),
                "primary_auth_failure_reason": reason,
                "tags": ["model-fallback", "primary-auth-failure"],
            }
        )
        return request.override(state=state)

    async def awrap_model_call(self, request, handler):
        """Run the model and fall back only for non-authentication errors."""
        try:
            return await handler(request)
        except Exception as exception:
            reason = classify_provider_authentication_error(exception)
            self._handle_primary_error(exception, request)
            last_exception = exception
        for fallback_model in self.models:
            fallback_request = self._fallback_request(request, fallback_model, reason)
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except Exception as exception:
                last_exception = exception
        raise last_exception

    def wrap_model_call(self, request, handler):
        """Run the model and fall back only for non-authentication errors."""
        try:
            return handler(request)
        except Exception as exception:
            reason = classify_provider_authentication_error(exception)
            self._handle_primary_error(exception, request)
            last_exception = exception
        for fallback_model in self.models:
            fallback_request = self._fallback_request(request, fallback_model, reason)
            try:
                return handler(fallback_request.override(model=fallback_model))
            except Exception as exception:
                last_exception = exception
        raise last_exception


# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    def invoke(self, input, config=None, **kwargs):
        """Invoke without retrying or falling back on auth failures."""
        try:
            return super().invoke(input, config=config, **kwargs)
        except Exception as exception:
            if (
                is_provider_authentication_error(exception)
                and is_strict_authentication_validation()
            ):
                raise ProviderAuthenticationError(
                    classify_provider_authentication_error(exception)
                ) from exception
            raise

    async def ainvoke(self, input, config=None, **kwargs):
        """Invoke asynchronously without retrying auth failures."""
        try:
            return await super().ainvoke(input, config=config, **kwargs)
        except Exception as exception:
            if (
                is_provider_authentication_error(exception)
                and is_strict_authentication_validation()
            ):
                raise ProviderAuthenticationError(
                    classify_provider_authentication_error(exception)
                ) from exception
            raise

    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: (
                not isinstance(exception, ValueError)
                and not is_provider_authentication_error(exception)
            )
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
                if isinstance(e, ValueError):
                    raise
                if is_provider_authentication_error(e):
                    if is_strict_authentication_validation():
                        raise ProviderAuthenticationError(
                            classify_provider_authentication_error(e)
                        ) from e
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
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "ProviderAuthenticationError",
    "ProviderAwareModelFallbackMiddleware",
    "classify_provider_authentication_error",
    "is_provider_authentication_error",
    "is_strict_authentication_validation",
]
