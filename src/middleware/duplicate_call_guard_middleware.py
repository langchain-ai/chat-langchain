"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
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
        seen_calls, check_links_count = self._turn_ledger(request)
        node_cache = self._node_cache(request)
        seen_calls.update(node_cache)
        check_links_count = sum(call_key[0] == "check_links" for call_key in seen_calls)

        if tool_name == "check_links" and check_links_count >= 1:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        result = await handler(request)
        if isinstance(result, ToolMessage) and result.status == "success":
            node_cache[call_key] = self._content_text(result.content)
        return result

    def _turn_ledger(
        self, request: ToolCallRequest
    ) -> tuple[dict[tuple[str, str], str], int]:
        messages = self._messages(request.state)
        turn_start = 0
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, HumanMessage)
                or getattr(message, "type", None) == "human"
            ):
                turn_start = index
                break
        current_turn = messages[turn_start:]
        tool_messages = {
            message.tool_call_id: message
            for message in current_turn
            if isinstance(message, ToolMessage)
        }
        ledger = {}
        for message in current_turn:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                tool_message = tool_messages.get(tool_call.get("id"))
                if tool_message is None or self._is_error(tool_message):
                    continue
                call_key = (
                    str(tool_call.get("name", "unknown_tool")),
                    self._canonical_args(tool_call.get("args", {})),
                )
                ledger[call_key] = self._content_text(tool_message.content)
        check_links_count = sum(call_key[0] == "check_links" for call_key in ledger)
        return ledger, check_links_count

    def _node_cache(self, request: ToolCallRequest) -> dict[tuple[str, str], str]:
        state = request.state
        if isinstance(state, MutableMapping):
            caches = state.setdefault("_duplicate_call_guard_node_cache", {})
            if isinstance(caches, dict):
                turn_key = self._turn_key(self._messages(state))
                cache = caches.setdefault(turn_key, {})
                if isinstance(cache, dict):
                    return cache
        return {}

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, HumanMessage)
                or getattr(message, "type", None) == "human"
            ):
                return f"{index}:{getattr(message, 'id', None)}:{message.content!r}"
        return "no-human-message"

    def _is_error(self, message: ToolMessage) -> bool:
        return str(message.status).lower() in {"error", "failure", "failed"}

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
