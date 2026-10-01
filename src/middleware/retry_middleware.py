"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables.retry import RunnableRetry
from langsmith.run_helpers import get_current_run_tree
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


def is_provider_auth_error(exception: BaseException) -> bool:
    """Return whether an exception indicates invalid provider credentials."""
    message = str(exception).upper()
    status_code = getattr(exception, "status_code", None)
    if status_code in (401, "401"):
        return True
    return "API_KEY_INVALID" in message or (
        "401" in message and ("AUTHENTICATION" in message or "UNAUTHORIZED" in message)
    )


def _record_fallback_metadata(error: BaseException) -> None:
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    error_kind = "auth" if is_provider_auth_error(error) else "other"
    run_tree.add_metadata({"served_by_fallback": True, "primary_error": error_kind})
    run_tree.add_tags(["served_by_fallback", f"primary_error:{error_kind}"])
    run_tree.patch()


def _model_name(model: object) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "_llm_type", None)
        or type(model).__name__
    )


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: (
                not isinstance(exception, ValueError)
                and not is_provider_auth_error(exception)
            )
        )
        return kwargs


class ObservableModelFallbackMiddleware(ModelFallbackMiddleware):
    """Fallback to alternate models while recording primary model failures."""

    def wrap_model_call(self, request, handler):
        primary_failure: list[BaseException] = []

        def tracked_handler(current_request):
            try:
                return handler(current_request)
            except Exception as error:
                if not primary_failure:
                    primary_failure.append(error)
                raise

        response = super().wrap_model_call(request, tracked_handler)
        if primary_failure:
            error = primary_failure[0]
            error_kind = "auth" if is_provider_auth_error(error) else "other"
            logger.error(
                "Primary model %s failed (%s): %s; serving fallback",
                _model_name(request.model),
                error_kind,
                error,
            )
            _record_fallback_metadata(error)
        return response

    async def awrap_model_call(self, request, handler):
        primary_failure: list[BaseException] = []

        async def tracked_handler(current_request):
            try:
                return await handler(current_request)
            except Exception as error:
                if not primary_failure:
                    primary_failure.append(error)
                raise

        response = await super().awrap_model_call(request, tracked_handler)
        if primary_failure:
            error = primary_failure[0]
            error_kind = "auth" if is_provider_auth_error(error) else "other"
            logger.error(
                "Primary model %s failed (%s): %s; serving fallback",
                _model_name(request.model),
                error_kind,
                error,
            )
            _record_fallback_metadata(error)
        return response


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
                if isinstance(e, ValueError) or is_provider_auth_error(e):
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
