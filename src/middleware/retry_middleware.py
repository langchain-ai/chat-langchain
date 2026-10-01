"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables.retry import RunnableRetry
from langsmith import get_current_run_tree
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}

AUTH_FAILURE_COOLDOWN = float(os.getenv("MODEL_AUTH_FAILURE_COOLDOWN", "300"))
_auth_failure_until: dict[str, float] = {}


def _exception_text(exception: BaseException) -> str:
    return str(exception).lower()


def is_auth_error(exception: BaseException) -> bool:
    """Return whether an exception indicates invalid provider credentials."""
    text = _exception_text(exception)
    if "api_key_invalid" in text or "api key not valid" in text:
        return True

    status_codes = {
        getattr(exception, "status_code", None),
        getattr(getattr(exception, "response", None), "status_code", None),
        getattr(exception, "code", None),
    }
    if status_codes & {401, 403, "401", "403"}:
        return True

    class_name = exception.__class__.__name__.lower()
    module_name = exception.__class__.__module__.lower()
    if class_name in {"permissiondenied", "unauthenticated", "authenticationerror"}:
        return True
    return "authenticationerror" in class_name or (
        "google.api_core" in module_name
        and class_name in {"permissiondenied", "unauthenticated"}
    )


def provider_model_key(model: object) -> str:
    """Return a stable provider/model key for breaker state."""
    if isinstance(model, str):
        return model
    for attribute in ("model_name", "model", "name"):
        value = getattr(model, attribute, None)
        if value:
            return str(value)
    return type(model).__name__


class ProviderCircuitOpenError(RuntimeError):
    """Raised when a provider is cooling down after an authentication failure."""


def is_provider_circuit_open(model: object) -> bool:
    """Return whether a provider/model is within its auth-failure cooldown."""
    return _auth_failure_until.get(provider_model_key(model), 0) > time.monotonic()


def record_auth_failure(model: object, exception: BaseException) -> None:
    """Open the provider/model circuit after an authentication failure."""
    key = provider_model_key(model)
    _auth_failure_until[key] = time.monotonic() + AUTH_FAILURE_COOLDOWN
    logger.error("Authentication failed for model %s; circuit opened: %s", key, exception)


def clear_provider_circuit(model: object) -> None:
    """Close a provider/model circuit after a successful call."""
    _auth_failure_until.pop(provider_model_key(model), None)


def reset_provider_circuits() -> None:
    """Clear all provider/model circuit state."""
    _auth_failure_until.clear()


def _response_model(response: ModelResponse, request: ModelRequest) -> str:
    for message in reversed(response.result):
        if isinstance(message, AIMessage):
            metadata = message.response_metadata or {}
            for key in ("model_name", "model", "model_id"):
                if metadata.get(key):
                    return str(metadata[key])
    return provider_model_key(request.model)


def _record_served_model(response: ModelResponse, request: ModelRequest) -> None:
    run_tree = get_current_run_tree()
    while run_tree and run_tree.parent_run:
        run_tree = run_tree.parent_run
    if run_tree:
        run_tree.metadata["served_model"] = _response_model(response, request)


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    provider_model: str | None = None

    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not isinstance(exception, ValueError)
            and not is_auth_error(exception)
        )
        return kwargs

    def _invoke(self, input_: Any, run_manager: Any, config: Any, **kwargs: Any) -> Any:
        model = self.provider_model or self.bound
        if is_provider_circuit_open(model):
            raise ProviderCircuitOpenError(f"Provider circuit open for {provider_model_key(model)}")
        try:
            result = super()._invoke(input_, run_manager, config, **kwargs)
        except Exception as exception:
            if is_auth_error(exception):
                record_auth_failure(model, exception)
            raise
        clear_provider_circuit(model)
        return result

    async def _ainvoke(
        self, input_: Any, run_manager: Any, config: Any, **kwargs: Any
    ) -> Any:
        model = self.provider_model or self.bound
        if is_provider_circuit_open(model):
            raise ProviderCircuitOpenError(f"Provider circuit open for {provider_model_key(model)}")
        try:
            result = await super()._ainvoke(input_, run_manager, config, **kwargs)
        except Exception as exception:
            if is_auth_error(exception):
                record_auth_failure(model, exception)
            raise
        clear_provider_circuit(model)
        return result


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
                if is_provider_circuit_open(request.model):
                    raise ProviderCircuitOpenError(
                        f"Provider circuit open for {provider_model_key(request.model)}"
                    )
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)
                _record_served_model(response, request)
                clear_provider_circuit(request.model)

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
                if isinstance(e, (ProviderCircuitOpenError, ValueError)):
                    raise
                if is_auth_error(e):
                    record_auth_failure(request.model, e)
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
    "AUTH_FAILURE_COOLDOWN",
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "ProviderCircuitOpenError",
    "_ProviderValidationAwareRunnableRetry",
    "clear_provider_circuit",
    "is_auth_error",
    "is_provider_circuit_open",
    "provider_model_key",
    "record_auth_failure",
    "reset_provider_circuits",
]
