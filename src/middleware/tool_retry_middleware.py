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

RESULT_FAILURE_PREFIXES = (
    "search failed:",
    "docs filesystem query failed:",
    "tool call failed:",
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
    ):
        super().__init__()
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "unknown_tool")

    def _tool_call_id(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("id", "")

    def _error_text(self, error: Exception) -> str:
        return str(error) or error.__class__.__name__

    def _status_code(self, error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        text = self._error_text(error)
        status_match = re.search(
            r"\b(?:HTTP|status(?:\s+code)?|error\s+code)[:= ]+"
            r"(429|500|502|503|504)\b",
            text,
            re.IGNORECASE,
        )
        if status_match:
            return int(status_match.group(1))

        return None

    def _is_no_results(self, error: Exception) -> bool:
        text = self._error_text(error).lower()
        return any(marker in text for marker in NO_RESULTS_MARKERS)

    def _is_retryable(self, error: Exception) -> bool:
        text = self._error_text(error).lower()
        status_code = self._status_code(error)
        if status_code in RETRYABLE_STATUS_CODES:
            return True

        return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)

    def _result_text(self, result: ToolMessage | Command) -> str:
        if isinstance(result, Command) or not isinstance(result, ToolMessage):
            return ""

        content = result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        return ""

    def _is_failure_result(self, result: ToolMessage | Command) -> bool:
        text = self._result_text(result).strip().lower()
        if not text:
            return False
        if text.startswith(RESULT_FAILURE_PREFIXES):
            return True
        if len(text) >= 400:
            return False

        status_code = re.search(r"(?<!\d)(?:429|500|502|503|504)(?!\d)", text)
        if status_code:
            return True

        for marker in RETRYABLE_ERROR_MARKERS:
            if marker in ("timeout", "timed out"):
                failure_context = re.search(
                    rf"\b(?:error|failed|failure|gateway|http|request|status|code|service|connection|retry|unavailable)\b.{{0,40}}\b{re.escape(marker)}\b",
                    text,
                )
                if failure_context:
                    return True
            elif marker in text:
                return True
        return False

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
        error: Exception,
    ) -> str:
        tool_name = self._tool_name(request)
        payload: dict[str, Any] = {
            "error": "Tool unavailable",
            "message": f"{tool_name} failed after {self.max_attempts} attempts.",
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
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await handler(request)
            except Exception as error:
                last_error = error
                tool_name = self._tool_name(request)

                if self._is_no_results(error):
                    logger.info(
                        "Tool %s returned no results; normalizing as tool output",
                        tool_name,
                    )
                    return self._tool_message(request, "No results found.")

                if self._is_retryable(error) and attempt < self.max_attempts:
                    delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
                    logger.warning(
                        "Tool %s failed attempt %s/%s: %s; retrying in %.2fs",
                        tool_name,
                        attempt,
                        self.max_attempts,
                        self._error_text(error),
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.warning(
                    "Tool %s failed after %s/%s attempts: %s",
                    tool_name,
                    attempt,
                    self.max_attempts,
                    self._error_text(error),
                )
                return self._tool_message(
                    request,
                    self._final_error_content(request, error),
                )

            if not self._is_failure_result(result):
                return result

            result_text = self._result_text(result)
            tool_name = self._tool_name(request)
            if attempt < self.max_attempts:
                delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
                logger.warning(
                    "Tool %s returned failure attempt %s/%s: %s; retrying in %.2fs",
                    tool_name,
                    attempt,
                    self.max_attempts,
                    result_text,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            logger.warning(
                "Tool %s returned failure after %s/%s attempts: %s",
                tool_name,
                attempt,
                self.max_attempts,
                result_text,
            )
            return self._tool_message(
                request,
                self._final_error_content(
                    request,
                    RuntimeError(result_text[:160]),
                ),
            )

        # Defensive fallback; loop should always return on success or final error.
        assert last_error is not None
        return self._tool_message(
            request, self._final_error_content(request, last_error)
        )


__all__ = ["ToolRetryMiddleware"]
