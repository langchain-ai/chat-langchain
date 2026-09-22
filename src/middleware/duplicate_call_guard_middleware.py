"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_STATE_KEY = "duplicate_call_guard"
_MAX_REPEATED_CALLS = 3
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
        seen_calls = turn_state["seen_calls"]

        call_key = self._call_key(tool_name, request.tool_call.get("args", {}))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            turn_state["repeated_call_count"] += 1
            if turn_state["repeated_call_count"] >= _MAX_REPEATED_CALLS:
                return self._termination_command(request)
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        if tool_name == "check_links" and turn_state["check_links_used"]:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        if tool_name == "check_links":
            turn_state["check_links_used"] = True

        result = await handler(request)
        if isinstance(result, ToolMessage) and result.status == "success":
            seen_calls[call_key] = self._content_text(result.content)
        return result

    def _turn_state(self, request: ToolCallRequest) -> dict[str, Any]:
        state = request.state
        if not isinstance(state, MutableMapping):
            return {
                "latest_human_index": self._latest_human_index(state),
                "seen_calls": {},
                "check_links_used": False,
                "repeated_call_count": 0,
            }
        latest_human_index = self._latest_human_index(state)
        current = state.get(_STATE_KEY)
        if (
            not isinstance(current, dict)
            or current.get("latest_human_index") != latest_human_index
        ):
            current = {
                "latest_human_index": latest_human_index,
                "seen_calls": {},
                "check_links_used": False,
                "repeated_call_count": 0,
            }
            state[_STATE_KEY] = current
        return current

    def _latest_human_index(self, state: Any) -> int:
        messages = self._messages(state)
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

    def _call_key(self, tool_name: str, args: Any) -> str:
        return f"{tool_name}\0{self._canonical_args(args)}"

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

    def _termination_command(self, request: ToolCallRequest) -> Command:
        answer = ""
        validated_links: list[str] = []
        for message in reversed(self._messages(request.state)):
            if isinstance(message, ToolMessage):
                if message.name == "check_links" and message.status == "success":
                    validated_links.append(self._content_text(message.content))
            elif isinstance(message, AIMessage) and not answer:
                content = self._content_text(message.content).strip()
                if content:
                    answer = content
        if not answer:
            answer = "This turn was stopped after repeated tool calls."
        if validated_links:
            answer += "\n\nValidated links:\n" + "\n".join(reversed(validated_links))
        return Command(update={"messages": [AIMessage(content=answer)]}, goto=END)
