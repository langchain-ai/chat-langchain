"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_TurnState = tuple[dict[tuple[str, str], str], set[str]]
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
        turn_state = self._turn_state(request)
        seen_calls = turn_state[0]

        if tool_name == "check_links" and "check_links" in turn_state[1]:
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

    def _turn_state(self, request: ToolCallRequest) -> _TurnState:
        messages = self._messages(request.state)
        latest_human_index = -1
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage) or getattr(
                messages[index], "type", None
            ) == "human":
                latest_human_index = index
                break

        turn_messages = messages[latest_human_index + 1 :]
        successful_results = {
            message.tool_call_id: message
            for message in turn_messages
            if isinstance(message, ToolMessage) and message.status == "success"
        }
        seen_calls: dict[tuple[str, str], str] = {}
        check_links_calls: set[str] = set()
        for message in turn_messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                result = successful_results.get(tool_call.get("id", ""))
                if result is None:
                    continue
                tool_name = str(tool_call.get("name", "unknown_tool"))
                call_key = (
                    tool_name,
                    self._canonical_args(tool_call.get("args", {})),
                )
                seen_calls[call_key] = self._content_text(result.content)
                if tool_name == "check_links":
                    check_links_calls.add("check_links")
        return seen_calls, check_links_calls

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
