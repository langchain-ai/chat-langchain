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

# Sentinel content some dynamically-bound tools leak on error; left as-is the
# LLM reads "nothing" as a successful empty result and fabricates confirmations.
EMPTY_ERROR_CONTENT_MARKERS = ("", "nothing", "none", "null")

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

    def _tool_message(
        self,
        request: ToolCallRequest,
        content: str,
        status: str = "success",
    ) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=self._tool_name(request),
            tool_call_id=self._tool_call_id(request),
            status=status,
        )

    def _normalize_error_message(self, message: ToolMessage) -> ToolMessage:
        """Rewrite ambiguous error ToolMessages so the LLM cannot read them as success."""
        if getattr(message, "status", None) != "error":
            return message

        content = message.content
        text = content if isinstance(content, str) else ""
        stripped = text.strip().lower()
        tool_name = getattr(message, "name", None) or "tool"

        if stripped in EMPTY_ERROR_CONTENT_MARKERS:
            message.content = (
                f"[TOOL ERROR] {tool_name} failed: no error detail returned "
                "\u2014 the action did NOT complete."
            )
        elif not stripped.startswith("[tool error]"):
            message.content = f"[TOOL ERROR] {text}"
        return message

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
        return f"[TOOL ERROR] {json.dumps(payload)}"

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
                    delay = self.initial_delay * (
                        self.backoff_factor ** (attempt - 1)
                    )
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
                    status="error",
                )
            else:
                if isinstance(result, ToolMessage):
                    return self._normalize_error_message(result)
                return result

        # Defensive fallback; loop should always return on success or final error.
        assert last_error is not None
        return self._tool_message(
            request,
            self._final_error_content(request, last_error),
            status="error",
        )


__all__ = ["ToolRetryMiddleware"]
