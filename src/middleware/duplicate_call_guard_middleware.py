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


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        messages = self._turn_messages(request.state)
        successful_results = {
            message.tool_call_id: message
            for message in messages
            if isinstance(message, ToolMessage) and message.status == "success"
        }
        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                if tool_name == "check_links" and tool_call["name"] == "check_links":
                    if tool_call["id"] != request.tool_call.get("id"):
                        return self._tool_message(request, _CHECK_LINKS_REFUSAL)
                    return await handler(request)
                if tool_call["id"] == request.tool_call.get("id"):
                    continue
                prior_key = (tool_call["name"], self._canonical_args(tool_call["args"]))
                result = successful_results.get(tool_call["id"] or "")
                if prior_key == call_key and result is not None:
                    return self._tool_message(
                        request,
                        f"{_DUPLICATE_NOTE}\n{self._content_text(result.content)}",
                    )
        return await handler(request)

    def _turn_messages(self, state: Any) -> list[BaseMessage]:
        messages = self._messages(state)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, HumanMessage)
                or getattr(message, "type", None) == "human"
            ):
                return messages[index + 1 :]
        return messages

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, Mapping):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", [])
        return list(messages or [])

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
