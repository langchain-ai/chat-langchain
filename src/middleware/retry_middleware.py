# Retry middleware for model calls with exponential backoff.
#
# Provider-agnostic: classifies transient upstream errors (5xx, 429,
# Anthropic OverloadedError, Google ChatGoogleGenerativeAIError, etc.)
# and retries with exponential backoff + jitter. On exhaustion, emits a
# user-facing assistant message instead of raising -- otherwise the root
# run terminates with empty outputs and the user/scheduler gets no signal.
# For cron-triggered runs, additionally logs a structured failure event
# the scheduler can react to.
import asyncio
import logging
import random
import re
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}

# HTTP status codes that we consider transient and worth retrying.
# 429 (rate limit) is retried even though it's 4xx -- it's a deterministic
# server-side signal to back off, not a client bug. All other 4xx are
# deterministic client errors (auth, bad request, etc.) and are NOT retried.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}

# Substrings that indicate a transient upstream failure when we can't get
# a status code off the exception directly (e.g. provider SDKs that wrap
# the HTTP error in their own exception class with the status in the message).
RETRYABLE_ERROR_MARKERS = (
    "bad gateway",
    "connection error",
    "connection reset",
    "gateway time-out",
    "gateway timeout",
    "internalservererror",
    "internal server error",
    "overloadederror",
    "overloaded",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
    "rate limit",
    "ratelimiterror",
    "api_error",  # Anthropic surfaces transient upstream errors as type=api_error
)

# Provider-specific exception class names known to indicate transient
# upstream failures. Matched by name to avoid importing every provider
# SDK at module load time.
RETRYABLE_EXCEPTION_NAMES = {
    # Anthropic
    "InternalServerError",
    "OverloadedError",
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    # Google
    "ChatGoogleGenerativeAIError",
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    # OpenAI / generic
    "APIStatusError",
    # httpx / requests
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
}

USER_FACING_FALLBACK_MESSAGE = (
    "The model is temporarily unavailable. Please retry in a moment."
)


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


def _status_code_from_exception(error: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from a provider exception."""
    # Many SDKs (OpenAI, Anthropic, httpx) expose .status_code directly.
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    # Some wrap an httpx Response on .response.
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    # google-api-core surfaces .code() returning an enum with .value.
    code_attr = getattr(error, "code", None)
    if callable(code_attr):
        try:
            value = code_attr()
            value = getattr(value, "value", value)
            if isinstance(value, int):
                return value
        except Exception:
            pass
    elif isinstance(code_attr, int):
        return code_attr

    # Last resort: scrape the message for "status 503" / "HTTP 500" / "code: 429".
    text = str(error) or ""
    status_match = re.search(
        r"\b(?:HTTP|status(?:\s+code)?|error\s+code|code)[:= ]+"
        r"(\d{3})\b",
        text,
        re.IGNORECASE,
    )
    if status_match:
        try:
            return int(status_match.group(1))
        except ValueError:
            return None
    return None


def _is_transient_provider_error(error: Exception) -> bool:
    """Return True if `error` looks like a transient upstream provider failure.

    Retried: 5xx, 429, OverloadedError (529), provider-SDK transient classes.
    NOT retried: 4xx other than 429 -- those are deterministic client errors
    (auth, schema, content filter) and retrying won't help.
    """
    status_code = _status_code_from_exception(error)

    # Status-code match has highest priority -- a 400 APIStatusError must
    # NOT be retried even though APIStatusError is in the transient class
    # list (it's the generic OpenAI base class).
    if status_code is not None:
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        if 400 <= status_code < 500:
            # 4xx other than 429 -- deterministic client error.
            return False
        if status_code >= 500:
            return True

    # Exception class name match (covers Anthropic InternalServerError,
    # Google ChatGoogleGenerativeAIError, etc. without importing them).
    for cls in type(error).mro():
        if cls.__name__ in RETRYABLE_EXCEPTION_NAMES:
            return True

    # Fallback: substring match against the error message.
    text = str(error).lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def _extract_run_metadata(request: ModelRequest) -> dict[str, Any]:
    """Pull a flat metadata dict off the model request's runtime config.

    Searches both `config["metadata"]` and `config["configurable"]` and
    flattens any nested `custom_metadata` sub-dict so callers can look up
    `source`, `cron_id`, `thread_id`, `run_id` uniformly.
    """
    out: dict[str, Any] = {}
    runtime = getattr(request, "runtime", None)
    if runtime is None:
        return out

    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        return out

    for key in ("metadata", "configurable"):
        raw = config.get(key)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k == "custom_metadata" and isinstance(v, dict):
                    for ck, cv in v.items():
                        out.setdefault(ck, cv)
                else:
                    out.setdefault(k, v)
    return out


def _is_cron_triggered(metadata: dict[str, Any]) -> bool:
    """Detect whether the current run was triggered by a cron/scheduler."""
    if metadata.get("source") == "trigger":
        return True
    if metadata.get("cron_id"):
        return True
    return False


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient upstream model errors with exponential backoff + jitter.

    On final failure, emits a user-facing assistant message instead of letting
    the exception terminate the root run with empty outputs. For cron-triggered
    runs, additionally logs a structured failure event so the scheduler has a
    signal to act on.
    """

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.25,
        fallback_message: str = USER_FACING_FALLBACK_MESSAGE,
    ):
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.fallback_message = fallback_message

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        metadata = getattr(response, "response_metadata", None) or {}
        return metadata.get("finish_reason", "")

    def _sleep_for(self, attempt: int) -> float:
        """Exponential backoff with bounded jitter."""
        base = self.initial_delay * (self.backoff_factor**attempt)
        if self.jitter > 0:
            base = base + random.uniform(0, self.jitter * base)
        return base

    def _log_cron_failure(
        self,
        error: Exception | None,
        metadata: dict[str, Any],
        attempts: int,
    ) -> None:
        """Emit a structured failure event for cron-triggered runs.

        Logs at ERROR level with a structured `extra` payload so log
        sinks (Datadog, Cloud Logging) can route on
        `event=cron_model_failure` and the scheduler can correlate by
        cron_id / thread_id.
        """
        payload = {
            "event": "cron_model_failure",
            "cron_id": metadata.get("cron_id"),
            "source": metadata.get("source"),
            "thread_id": metadata.get("thread_id"),
            "run_id": metadata.get("run_id"),
            "error_type": type(error).__name__ if error else None,
            "error": str(error)[:500] if error else None,
            "attempts": attempts,
        }
        logger.error(
            "Cron-triggered model call failed after %s attempts: %s",
            attempts,
            error,
            extra={"structured": payload},
        )

    def _fallback_response(self) -> dict[str, Any]:
        """Build a state-update returning a user-facing AIMessage.

        Returning a dict from awrap_model_call short-circuits the model
        call and merges the dict into agent state -- this is how we emit
        a user-visible message instead of letting the exception bubble
        up and terminate the run with empty outputs.
        """
        return {
            "messages": [AIMessage(content=self.fallback_message)],
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        last_exception: Exception | None = None
        last_retryable_reason: str | None = None
        total_attempts = self.max_retries + 1

        for attempt in range(total_attempts):
            try:
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)

                if finish_reason in RETRYABLE_FINISH_REASONS:
                    if attempt < self.max_retries:
                        delay = self._sleep_for(attempt)
                        logger.warning(
                            f"Retryable response ({finish_reason}) "
                            f"attempt {attempt + 1}/{total_attempts}, "
                            f"retrying in {delay:.2f}s"
                        )
                        last_retryable_reason = finish_reason
                        await asyncio.sleep(delay)
                        continue

                return response

            except Exception as e:
                last_exception = e
                transient = _is_transient_provider_error(e)

                if not transient:
                    # Deterministic client error (4xx other than 429,
                    # auth failures, schema errors). Don't retry -- let
                    # downstream middleware (e.g. ErrorNormalization) see it.
                    logger.warning(
                        f"Non-retryable model error on attempt "
                        f"{attempt + 1}/{total_attempts}: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise

                if attempt < self.max_retries:
                    delay = self._sleep_for(attempt)
                    logger.warning(
                        f"Transient model error attempt "
                        f"{attempt + 1}/{total_attempts}: "
                        f"{type(e).__name__}: {e}, retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Model call failed after {total_attempts} attempts: "
                        f"{type(e).__name__}: {e}"
                    )

        # Exhausted retries on a transient error. Emit a user-facing fallback
        # message instead of raising -- otherwise the root run terminates with
        # empty outputs and neither the user nor the scheduler gets a signal.
        if last_exception is not None:
            metadata = _extract_run_metadata(request)
            if _is_cron_triggered(metadata):
                self._log_cron_failure(last_exception, metadata, total_attempts)
            return self._fallback_response()

        if last_retryable_reason:
            # Malformed-response path (no exception, but final attempt also
            # came back with a retryable finish_reason). Surface the raw
            # error so existing tests/normalization can act on it.
            raise MalformedResponseError(
                f"Model returned {last_retryable_reason} after "
                f"{total_attempts} attempts"
            )

        raise RuntimeError("Unexpected state in retry middleware")


__all__ = [
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "USER_FACING_FALLBACK_MESSAGE",
]
