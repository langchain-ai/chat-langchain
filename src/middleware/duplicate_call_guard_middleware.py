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
        seen_calls, check_links_calls = self._turn_state(request)

        if tool_name == "check_links" and check_links_calls:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        result = await handler(request)
        return result

    def _turn_state(
        self, request: ToolCallRequest
    ) -> tuple[dict[tuple[str, str], str], int]:
        messages = self._messages(request.state)
        latest_human_index = self._latest_human_index(messages)
        pending_calls: dict[str, tuple[str, str]] = {}
        seen_calls: dict[tuple[str, str], str] = {}
        check_links_calls = 0
        for message in messages[latest_human_index + 1 :]:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    pending_calls[tool_call["id"]] = (
                        str(tool_call.get("name", "unknown_tool")),
                        self._canonical_args(tool_call.get("args", {})),
                    )
            elif isinstance(message, ToolMessage) and message.status == "success":
                tool_call = pending_calls.get(message.tool_call_id)
                if tool_call is None:
                    continue
                seen_calls[tool_call] = self._content_text(message.content)
                if tool_call[0] == "check_links":
                    check_links_calls += 1
        return seen_calls, check_links_calls

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
                return index
        return -1

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, Mapping):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", state or [])
        return list(messages)

    def _canonical_args(self, args: Any) -> str:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

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
