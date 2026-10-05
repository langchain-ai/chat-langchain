"""Retry and sanitize tool-call failures before they reach users."""
import asyncio
import json
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from src.tools.pylon_tools import PylonUnavailableError

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
    ):
        """Initialize retry settings and the compiled tool registry."""
        super().__init__()
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self._tools_by_name: dict[str, BaseTool] = {}

    def _set_tool_registry(self, tools: list[BaseTool] | None) -> None:
        if tools:
            self._tools_by_name = {tool.name: tool for tool in tools}

    def _registered_tool_names(self) -> list[str]:
        return list(self._tools_by_name)

    def _resolve_tool_name(self, name: str) -> str | None:
        if name in self._tools_by_name:
            return name

        cleaned_name = "".join(character for character in name if character.isprintable())
        trailing_name = cleaned_name.rsplit(":", 1)[-1]
        if trailing_name in self._tools_by_name:
            return trailing_name

        return next(
            (
                registered_name
                for registered_name in sorted(
                    self._tools_by_name, key=len, reverse=True
                )
                if registered_name in cleaned_name
            ),
            None,
        )

    def _prune_invalid_tool_calls(self, state: AgentState) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not self._tools_by_name:
            return None

        updates: list[Any] = []
        tool_messages_by_id = {
            message.tool_call_id: message
            for message in messages
            if isinstance(message, ToolMessage)
        }
        for message in messages:
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue

            valid_tool_calls = [
                tool_call
                for tool_call in message.tool_calls
                if tool_call.get("name") in self._tools_by_name
            ]
            invalid_tool_calls = [
                tool_call
                for tool_call in message.tool_calls
                if tool_call.get("name") not in self._tools_by_name
            ]
            if not invalid_tool_calls or message.id is None:
                continue

            updates.append(RemoveMessage(id=message.id))
            updates.extend(
                RemoveMessage(id=tool_messages_by_id[tool_call["id"]].id)
                for tool_call in invalid_tool_calls
                if tool_call.get("id") in tool_messages_by_id
            )
            updates.append(message.model_copy(update={"tool_calls": valid_tool_calls}))

        return {"messages": updates} if updates else None

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
        """Normalize tool names and retry transient execution failures."""
        runtime_tools = getattr(request.runtime, "tools", None)
        self._set_tool_registry(runtime_tools)
        if not runtime_tools and request.tool is not None:
            self._set_tool_registry([request.tool])

        requested_name = request.tool_call.get("name", "")
        resolved_name = self._resolve_tool_name(requested_name)
        if resolved_name is not None and resolved_name != requested_name:
            request = request.override(
                tool_call={**request.tool_call, "name": resolved_name},
                tool=self._tools_by_name[resolved_name],
            )
        elif resolved_name is None and self._tools_by_name:
            return ToolMessage(
                content=(
                    "Unknown tool. Use one of: "
                    + ", ".join(self._registered_tool_names())
                    + "."
                ),
                name="tool_error",
                tool_call_id=self._tool_call_id(request),
                status="error",
            )

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await handler(request)
            except PylonUnavailableError as error:
                logger.warning(
                    "Tool %s unavailable: %s",
                    self._tool_name(request),
                    self._error_text(error),
                )
                return self._tool_message(
                    request,
                    self._error_text(error),
                    status="error",
                )
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
                )

        # Defensive fallback; loop should always return on success or final error.
        assert last_error is not None
        return self._tool_message(request, self._final_error_content(request, last_error))

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Remove invalid checkpointed tool calls before model execution."""
        return self._prune_invalid_tool_calls(state)


__all__ = ["ToolRetryMiddleware"]
