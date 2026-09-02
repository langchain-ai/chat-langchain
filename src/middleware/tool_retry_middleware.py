"""Retry and sanitize tool-call failures before they reach users."""

import asyncio
import contextvars
import json
import logging
import re
import time
from dataclasses import dataclass
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

RETRIEVAL_TOOL_NAMES = {
    "search_docs_by_lang_chain",
    "query_docs_filesystem_docs_by_lang_chain",
}
RETRIEVAL_FAILURE_LIMIT = 3
RETRIEVAL_TIME_LIMIT_SECONDS = 90.0

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


@dataclass
class _RetrievalTurnState:
    failed_calls: int = 0
    failed_seconds: float = 0.0


class ToolRetryMiddleware(AgentMiddleware[AgentState]):
    """Retry transient tool failures and return model-readable errors."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        """Configure retry attempts and exponential backoff."""
        super().__init__()
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self._retrieval_turn_state: contextvars.ContextVar[
            _RetrievalTurnState | None
        ] = contextvars.ContextVar("retrieval_turn_state", default=None)

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "unknown_tool")

    def _tool_call_id(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("id", "")

    def before_agent(self, state: AgentState, runtime: Any) -> None:
        """Initialize retrieval failure state for the agent turn."""
        self._retrieval_turn_state.set(_RetrievalTurnState())

    def after_agent(self, state: AgentState, runtime: Any) -> None:
        """Clear retrieval failure state after the agent turn."""
        self._retrieval_turn_state.set(None)

    def _error_text(self, error: Exception) -> str:
        return str(error) or error.__class__.__name__

    def _result_text(self, result: ToolMessage | Command) -> str:
        content = getattr(result, "content", result)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_blocks = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and "text" in block
            ]
            if text_blocks:
                return "".join(str(text) for text in text_blocks)
        return str(content)

    def _status_code_from_text(self, text: str) -> int | None:
        status_match = re.search(
            r"\b(?:HTTP|status(?:\s+code)?|error\s+code)[:= ]+"
            r"(429|500|502|503|504)\b",
            text,
            re.IGNORECASE,
        )
        if status_match:
            return int(status_match.group(1))

        status_match = re.search(
            r"\b(429|500|502|503|504)\b(?=\s+(?:bad\s+gateway|gateway))",
            text,
            re.IGNORECASE,
        )
        if status_match:
            return int(status_match.group(1))
        return None

    def _is_retryable_text(self, text: str) -> bool:
        lowered = text.lower()
        if any(marker in lowered for marker in NO_RESULTS_MARKERS):
            return False
        return self._status_code_from_text(text) in RETRYABLE_STATUS_CODES or any(
            marker in lowered for marker in RETRYABLE_ERROR_MARKERS
        )

    def _status_code(self, error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        return self._status_code_from_text(self._error_text(error))

    def _is_no_results(self, error: Exception) -> bool:
        text = self._error_text(error).lower()
        return any(marker in text for marker in NO_RESULTS_MARKERS)

    def _is_retryable(self, error: Exception) -> bool:
        return self._status_code(
            error
        ) in RETRYABLE_STATUS_CODES or self._is_retryable_text(self._error_text(error))

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
    ) -> str:
        tool_name = self._tool_name(request)
        error_text = (
            self._error_text(error) if isinstance(error, Exception) else str(error)
        )
        payload: dict[str, Any] = {
            "error": "Tool unavailable",
            "message": f"{tool_name} failed after {self.max_attempts} attempts.",
            "tool": tool_name,
            "suggestion": (
                "Try a narrower or related query, use another available source, "
                "or answer from already retrieved context."
            ),
            "details": error_text[:160],
        }
        return json.dumps(payload)

    def _retrieval_state(self) -> _RetrievalTurnState:
        state = self._retrieval_turn_state.get()
        if state is None:
            state = _RetrievalTurnState()
            self._retrieval_turn_state.set(state)
        return state

    def _retrieval_budget_exhausted(self, tool_name: str) -> bool:
        return tool_name in RETRIEVAL_TOOL_NAMES and (
            (state := self._retrieval_state()).failed_calls >= RETRIEVAL_FAILURE_LIMIT
            or state.failed_seconds >= RETRIEVAL_TIME_LIMIT_SECONDS
        )

    def _record_retrieval_failure(
        self, tool_name: str, elapsed: float, completed_call: bool
    ) -> None:
        if tool_name in RETRIEVAL_TOOL_NAMES:
            state = self._retrieval_state()
            state.failed_seconds += elapsed
            if completed_call:
                state.failed_calls += 1

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Retry transient tool failures and normalize exhausted attempts."""
        tool_name = self._tool_name(request)
        if self._retrieval_budget_exhausted(tool_name):
            return self._tool_message(
                request,
                self._final_error_content(
                    request, "Documentation retrieval budget exhausted."
                ),
            )

        last_error: Exception | str | None = None

        for attempt in range(1, self.max_attempts + 1):
            started_at = time.monotonic()
            try:
                result = await handler(request)
            except Exception as error:
                elapsed = time.monotonic() - started_at
                last_error = error
                if self._is_no_results(error):
                    logger.info(
                        "Tool %s returned no results; normalizing as tool output",
                        tool_name,
                    )
                    return self._tool_message(request, "No results found.")
                retryable = self._is_retryable(error)
                if retryable:
                    self._record_retrieval_failure(
                        tool_name, elapsed, attempt >= self.max_attempts
                    )
                if (
                    retryable
                    and attempt < self.max_attempts
                    and not self._retrieval_budget_exhausted(tool_name)
                ):
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
                    request, self._final_error_content(request, error)
                )

            result_text = self._result_text(result)
            if not self._is_retryable_text(result_text):
                return result
            last_error = result_text
            elapsed = time.monotonic() - started_at
            self._record_retrieval_failure(
                tool_name, elapsed, attempt >= self.max_attempts
            )
            if attempt < self.max_attempts and not self._retrieval_budget_exhausted(
                tool_name
            ):
                delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
                logger.warning(
                    "Tool %s returned retryable content on attempt %s/%s: %s; retrying in %.2fs",
                    tool_name,
                    attempt,
                    self.max_attempts,
                    result_text,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            logger.warning(
                "Tool %s returned retryable content after %s/%s attempts: %s",
                tool_name,
                attempt,
                self.max_attempts,
                result_text,
            )
            return self._tool_message(
                request, self._final_error_content(request, result_text)
            )

        assert last_error is not None
        return self._tool_message(
            request, self._final_error_content(request, last_error)
        )


__all__ = ["ToolRetryMiddleware"]
