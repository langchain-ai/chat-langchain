"""Retry and sanitize tool-call failures before they reach users."""
import asyncio
import json
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

NO_RESULTS_MARKERS = (
    "no results found",
    "no result found",
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

RETRYABLE_ERROR_MARKERS = (
    "bad gateway",
    "connection error",
    "connection reset",
    "gateway time-out",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
)


class ToolRetryMiddleware(AgentMiddleware[AgentState]):
    """Retry transient tool failures and return model-readable errors."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        returned_error_attempts: int = 2,
    ):
        """Configure exception and returned-error retry budgets."""
        super().__init__()
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.returned_error_attempts = min(returned_error_attempts, max_attempts)

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "unknown_tool")

    def _tool_call_id(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("id", "")

    def _content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                text for text in (self._content_text(item) for item in content) if text
            )
        if isinstance(content, dict):
            for key in ("text", "content"):
                if key in content:
                    return self._content_text(content[key])
        return str(content)

    def _error_text(self, error: Exception | str) -> str:
        return str(error) or error.__class__.__name__

    def _status_code(self, error: Exception | str) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        text = self._error_text(error)
        status_codes = "|".join(str(code) for code in RETRYABLE_STATUS_CODES)
        status_match = re.search(
            rf"\b(?:HTTP|status(?:\s+code)?|error\s+code)?[:= ]*({status_codes})\b",
            text,
            re.IGNORECASE,
        )
        if status_match:
            return int(status_match.group(1))

        return None

    def _is_no_results(self, error: Exception | str) -> bool:
        text = self._error_text(error).lower()
        return any(marker in text for marker in NO_RESULTS_MARKERS)

    def _is_retryable(self, error: Exception | str) -> bool:
        text = self._error_text(error).lower()
        status_code = self._status_code(error)
        if status_code in RETRYABLE_STATUS_CODES:
            return True

        return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)

    def _is_error_result(self, result: Any) -> bool:
        if not isinstance(result, ToolMessage):
            return False
        text = self._content_text(result.content)
        return result.status == "error" or self._is_retryable(text)

    def _tool_message(
        self,
        request: ToolCallRequest,
        content: str,
    ) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=self._tool_name(request),
            tool_call_id=self._tool_call_id(request),
        )

    def _final_error_content(
        self,
        request: ToolCallRequest,
        error: Exception | str,
        attempts: int | None = None,
    ) -> str:
        tool_name = self._tool_name(request)
        attempt_count = self.max_attempts if attempts is None else attempts
        payload: dict[str, Any] = {
            "error": "Tool unavailable",
            "message": f"{tool_name} failed after {attempt_count} attempts.",
            "tool": tool_name,
            "suggestion": (
                "Try a narrower or related query, use another available source, "
                "or answer from already retrieved context."
            ),
            "details": self._error_text(error)[:160],
        }
        return json.dumps(payload)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Retry failed tool calls and sanitize their final output."""
        last_error: Exception | None = None
        exception_attempts = 0
        returned_error_attempts = 0

        while exception_attempts < self.max_attempts:
            try:
                result = await handler(request)
            except Exception as error:
                exception_attempts += 1
                last_error = error
                tool_name = self._tool_name(request)

                if self._is_no_results(error):
                    logger.info(
                        "Tool %s returned no results; normalizing as tool output",
                        tool_name,
                    )
                    return self._tool_message(request, "No results found.")

                if self._is_retryable(error) and exception_attempts < self.max_attempts:
                    delay = self.initial_delay * (
                        self.backoff_factor ** (exception_attempts - 1)
                    )
                    logger.warning(
                        "Tool %s failed attempt %s/%s: %s; retrying in %.2fs",
                        tool_name,
                        exception_attempts,
                        self.max_attempts,
                        self._error_text(error),
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.warning(
                    "Tool %s failed after %s/%s attempts: %s",
                    tool_name,
                    exception_attempts,
                    self.max_attempts,
                    self._error_text(error),
                )
                return self._tool_message(
                    request,
                    self._final_error_content(request, error),
                )

            if isinstance(result, ToolMessage):
                text = self._content_text(result.content)
                tool_name = self._tool_name(request)

                if self._is_no_results(text):
                    logger.info(
                        "Tool %s returned no results; normalizing as tool output",
                        tool_name,
                    )
                    return self._tool_message(request, "No results found.")

                if self._is_error_result(result):
                    returned_error_attempts += 1
                    if returned_error_attempts < self.returned_error_attempts:
                        delay = self.initial_delay * (
                            self.backoff_factor ** (returned_error_attempts - 1)
                        )
                        logger.warning(
                            "Tool %s returned error attempt %s/%s: %s; retrying in %.2fs",
                            tool_name,
                            returned_error_attempts,
                            self.returned_error_attempts,
                            text,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.warning(
                        "Tool %s returned error after %s/%s attempts: %s",
                        tool_name,
                        returned_error_attempts,
                        self.returned_error_attempts,
                        text,
                    )
                    return self._tool_message(
                        request,
                        self._final_error_content(
                            request,
                            text,
                            attempts=self.returned_error_attempts,
                        ),
                    )

            return result

        # Defensive fallback; loop should always return on success or final error.
        assert last_error is not None
        return self._tool_message(request, self._final_error_content(request, last_error))


__all__ = ["ToolRetryMiddleware"]
