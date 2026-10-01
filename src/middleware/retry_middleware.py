"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
from typing import Awaitable, Callable

from anthropic import AuthenticationError as AnthropicAuthenticationError
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import (
    _sanitize_request_for_fallback,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables.retry import RunnableRetry
from langchain_google_genai.chat_models import GoogleInvalidRequestError
from openai import AuthenticationError as OpenAIAuthenticationError
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}

_AUTHENTICATION_FAILURE_MODELS: set[str] = set()


def model_id(model: object) -> str:
    """Return the provider-qualified identifier for a model when available."""
    provider = getattr(model, "model_provider", None)
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if provider is None:
        provider = {
            "chat-google-generative-ai": "google_genai",
            "openai-chat": "openai",
            "anthropic-chat": "anthropic",
        }.get(getattr(model, "_llm_type", None))
    if provider and name:
        return f"{provider}:{name}"
    return str(name or getattr(model, "name", None) or model)


def is_authentication_failure(exception: BaseException) -> bool:
    """Identify provider errors caused by invalid or unauthorized credentials."""
    if isinstance(
        exception,
        (OpenAIAuthenticationError, AnthropicAuthenticationError),
    ):
        return True
    if isinstance(exception, GoogleInvalidRequestError) and "API_KEY_INVALID" in str(
        exception
    ):
        return True
    status_code = getattr(exception, "status_code", None)
    response = getattr(exception, "response", None)
    status_code = status_code or getattr(response, "status_code", None)
    return status_code in {401, 403}


def mark_model_authentication_failure(model: object) -> str:
    """Mark a model unavailable after a permanent authentication failure."""
    identifier = model_id(model)
    _AUTHENTICATION_FAILURE_MODELS.add(identifier)
    logger.error("Authentication failure for model %s", identifier)
    return identifier


def is_model_unavailable(model: object) -> bool:
    """Return whether a model has failed authentication in this process."""
    return model_id(model) in _AUTHENTICATION_FAILURE_MODELS


def record_model_outcome(model: object, *, fallback_used: bool) -> None:
    """Record the serving model on the active run when possible."""
    identifier = model_id(model)
    try:
        from langgraph.config import get_config

        config = get_config()
    except RuntimeError:
        config = None
    if config is not None:
        metadata = config.setdefault("metadata", {})
        metadata["served_model"] = identifier
        metadata["fallback_used"] = fallback_used
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
    except RuntimeError:
        run = None
    if run is not None:
        run.metadata["served_model"] = identifier
        run.metadata["fallback_used"] = fallback_used


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    model: str | None = None

    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: self._is_retryable(exception)
        )
        return kwargs

    def _is_retryable(self, exception: BaseException) -> bool:
        if is_authentication_failure(exception):
            if self.model:
                mark_model_authentication_failure(self.model)
            return False
        return not isinstance(exception, ValueError)


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
                if is_authentication_failure(e):
                    mark_model_authentication_failure(request.model)
                    raise
                if isinstance(e, ValueError):
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


class ModelFallbackWithCircuitBreakerMiddleware(ModelFallbackMiddleware):
    """Skip models that permanently failed authentication in this process."""

    def _record_auth_failure(self, exception: Exception, model: object) -> None:
        if is_authentication_failure(exception):
            mark_model_authentication_failure(model)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Run the primary model or the first available fallback."""
        primary_model = request.model
        last_exception: Exception | None = None
        if not is_model_unavailable(primary_model):
            try:
                response = handler(request)
                record_model_outcome(primary_model, fallback_used=False)
                return response
            except Exception as exception:
                last_exception = exception
                self._record_auth_failure(exception, primary_model)

        for fallback_model in self.models:
            if is_model_unavailable(fallback_model):
                continue
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                response = handler(fallback_request.override(model=fallback_model))
                record_model_outcome(fallback_model, fallback_used=True)
                return response
            except Exception as exception:
                last_exception = exception
                self._record_auth_failure(exception, fallback_model)

        if last_exception:
            raise last_exception
        raise RuntimeError("All configured models are unavailable")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Run the primary model or the first available fallback asynchronously."""
        primary_model = request.model
        last_exception: Exception | None = None
        if not is_model_unavailable(primary_model):
            try:
                response = await handler(request)
                record_model_outcome(primary_model, fallback_used=False)
                return response
            except Exception as exception:
                last_exception = exception
                self._record_auth_failure(exception, primary_model)

        for fallback_model in self.models:
            if is_model_unavailable(fallback_model):
                continue
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                response = await handler(
                    fallback_request.override(model=fallback_model)
                )
                record_model_outcome(fallback_model, fallback_used=True)
                return response
            except Exception as exception:
                last_exception = exception
                self._record_auth_failure(exception, fallback_model)

        if last_exception:
            raise last_exception
        raise RuntimeError("All configured models are unavailable")


__all__ = [
    "ModelFallbackWithCircuitBreakerMiddleware",
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "is_authentication_failure",
]
