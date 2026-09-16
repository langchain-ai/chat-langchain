"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)
_TOOL_CALL_CAP_REFUSAL = (
    "Stop calling this tool. Use the information already available to answer the user."
)


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce per-tool turn budgets."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        current_turn = self._current_turn_messages(request.state)
        tool_counts = self._tool_counts(current_turn)
        if tool_name == "check_links" and tool_counts.get(tool_name, 0) > 0:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = self._cached_successes(current_turn).get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        if tool_counts.get(tool_name, 0) >= 8:
            return self._tool_message(request, _TOOL_CALL_CAP_REFUSAL)

        result = await handler(request)
        return result

    def _current_turn_messages(self, state: Any) -> list[BaseMessage]:
        messages = self._messages(state)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, HumanMessage)
                or getattr(message, "type", None) == "human"
            ):
                return messages[index + 1 :]
        return messages

    def _tool_counts(self, messages: list[BaseMessage]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for message in messages:
            if isinstance(message, ToolMessage):
                tool_name = str(message.name or "unknown_tool")
                counts[tool_name] = counts.get(tool_name, 0) + 1
        return counts

    def _cached_successes(
        self, messages: list[BaseMessage]
    ) -> dict[tuple[str, str], str]:
        tool_messages = {
            message.tool_call_id: message
            for message in messages
            if isinstance(message, ToolMessage)
            and message.status == "success"
            and message.tool_call_id
        }
        cached: dict[tuple[str, str], str] = {}
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                tool_message = tool_messages.get(tool_call.get("id"))
                if tool_message is None:
                    continue
                tool_name = str(tool_call.get("name", "unknown_tool"))
                call_key = (tool_name, self._canonical_args(tool_call.get("args", {})))
                cached[call_key] = self._content_text(tool_message.content)
        return cached

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, Mapping):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", state or [])
        return list(messages)

    def _canonical_args(self, args: Any) -> str:
        return json.dumps(
            args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def _content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
        )
