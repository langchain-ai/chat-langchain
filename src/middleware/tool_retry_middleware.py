"""Retry and sanitize tool-call failures before they reach users."""

import asyncio
import json
import logging
import re
import time
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

DEFAULT_TOOL_TIMEOUTS = {
    "search_docs_by_lang_chain": 20.0,
    "query_docs_filesystem_docs_by_lang_chain": 20.0,
}


class ToolRetryMiddleware(AgentMiddleware[AgentState]):
    """Retry transient tool failures and return model-readable errors."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        tool_timeouts: dict[str, float] | None = None,
    ):
        """Configure retries and per-call timeouts for docs tools."""
        super().__init__()
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.tool_timeouts = dict(
            DEFAULT_TOOL_TIMEOUTS if tool_timeouts is None else tool_timeouts
        )

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
    ) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=self._tool_name(request),
            tool_call_id=self._tool_call_id(request),
        )

    def _content_error_text(self, result: ToolMessage | Command) -> str | None:
        if not isinstance(result, ToolMessage):
            return None
        content = result.content
        if isinstance(content, list):
            content = "\n".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
                if isinstance(block, str) or block.get("type") == "text"
            )
        text = content.strip()
        if text.startswith(("Search failed:", "Docs filesystem query failed:")):
            return text
        if len(text) < 300 and any(
            marker in text.lower() for marker in RETRYABLE_ERROR_MARKERS
        ):
            return text
        return None

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
        if tool_name in DEFAULT_TOOL_TIMEOUTS:
            payload["message"] = f"{tool_name} is unavailable."
            payload["suggestion"] = (
                "Use already retrieved documentation or acknowledge that "
                "documentation could not be found; do not make unsupported claims."
            )
            if tool_name == "search_docs_by_lang_chain":
                payload["suggestion"] = (
                    "Try query_docs_filesystem_docs_by_lang_chain to read the docs. "
                    + payload["suggestion"]
                )
        return json.dumps(payload)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Retry transient failures while bounding configured tool calls."""
        last_error: Exception | None = None
        tool_name = self._tool_name(request)
        timeout = self.tool_timeouts.get(tool_name)

        for attempt in range(1, self.max_attempts + 1):
            content_failure = False
            try:
                started = time.perf_counter()
                if timeout is None:
                    result = await handler(request)
                else:
                    result = await asyncio.wait_for(handler(request), timeout)
                if tool_name in DEFAULT_TOOL_TIMEOUTS:
                    text = self._content_error_text(result)
                    if text is not None:
                        error = RuntimeError(text)
                        if time.perf_counter() - started >= 5.0:
                            return self._tool_message(
                                request, self._final_error_content(request, error)
                            )
                        content_failure = True
                        raise error
                return result
            except Exception as error:
                last_error = error

                if timeout is not None and isinstance(error, asyncio.TimeoutError):
                    return self._tool_message(
                        request,
                        self._final_error_content(
                            request,
                            RuntimeError(
                                f"Tool call timed out after {timeout:g} seconds."
                            ),
                        ),
                    )

                if not content_failure and self._is_no_results(error):
                    logger.info(
                        "Tool %s returned no results; normalizing as tool output",
                        tool_name,
                    )
                    return self._tool_message(request, "No results found.")

                if (
                    content_failure or self._is_retryable(error)
                ) and attempt < self.max_attempts:
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

        # Defensive fallback; loop should always return on success or final error.
        assert last_error is not None
        return self._tool_message(
            request, self._final_error_content(request, last_error)
        )


__all__ = ["ToolRetryMiddleware"]
