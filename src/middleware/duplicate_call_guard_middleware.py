"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
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
        seen_calls, check_links_count = self._turn_history(request)

        if tool_name == "check_links" and check_links_count >= 1:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        return await handler(request)

    def _turn_history(
        self, request: ToolCallRequest
    ) -> tuple[dict[tuple[str, str], str], int]:
        messages = self._messages(request.state)
        latest_human_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if getattr(messages[index], "type", None) == "human"
            ),
            -1,
        )
        tool_calls = {}
        seen_calls = {}
        check_links_count = 0
        for message in messages[latest_human_index + 1 :]:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    tool_calls[tool_call["id"]] = (
                        str(tool_call.get("name", "unknown_tool")),
                        self._canonical_args(tool_call.get("args", {})),
                    )
            elif isinstance(message, ToolMessage) and message.status == "success":
                call_key = tool_calls.get(message.tool_call_id)
                if call_key is None:
                    continue
                seen_calls[call_key] = self._content_text(message.content)
                if call_key[0] == "check_links":
                    check_links_count += 1
        return seen_calls, check_links_count

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
