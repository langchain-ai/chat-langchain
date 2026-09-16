"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from src.middleware.citation_guard_middleware import (
    _RETRY_INSTRUCTIONS as _CITATION_RETRY_INSTRUCTIONS,
)
from src.middleware.docs_research_guard_middleware import (
    _RETRY_INSTRUCTIONS as _RESEARCH_RETRY_INSTRUCTIONS,
)

_TurnCallKey = tuple[str, str]


class _TurnState(TypedDict):
    calls: dict[_TurnCallKey, str]
    budget_markers: set[str]


class DuplicateCallGuardState(AgentState):
    """State schema for duplicate-call records."""

    duplicate_call_guard: NotRequired[dict[str, _TurnState]]


_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)
_SUMMARY_PREFIX = "Here is a summary of the conversation to date"


class DuplicateCallGuardMiddleware(AgentMiddleware[DuplicateCallGuardState]):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    state_schema = DuplicateCallGuardState

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        turn_key = f"{self._execution_key(request)}:{self._turn_key(request.state)}"
        persisted = self._persisted_state(request.state)
        turn_state = persisted.get(turn_key, {"calls": {}, "budget_markers": set()})
        seen_calls = turn_state["calls"]

        if tool_name == "check_links" and "check_links" in turn_state["budget_markers"]:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        updated_turn_state: _TurnState = {
            "calls": dict(seen_calls),
            "budget_markers": set(turn_state["budget_markers"]),
        }
        if tool_name == "check_links":
            updated_turn_state["budget_markers"].add("check_links")

        result = await handler(request)
        if not isinstance(result, ToolMessage) or result.status != "success":
            return result

        updated_turn_state["calls"][call_key] = self._content_text(result.content)
        updated_state = dict(persisted)
        updated_state[turn_key] = updated_turn_state
        return Command(
            update={
                "duplicate_call_guard": updated_state,
                "messages": [result],
            }
        )

    def _persisted_state(self, state: Any) -> dict[str, _TurnState]:
        if isinstance(state, Mapping):
            return dict(state.get("duplicate_call_guard", {}))
        return dict(getattr(state, "duplicate_call_guard", {}) or {})

    def _execution_key(self, request: ToolCallRequest) -> str:
        runtime = request.runtime
        config = getattr(runtime, "config", None)
        run_id = config.get("run_id") if isinstance(config, Mapping) else None
        if run_id:
            return f"run:{run_id}"
        return "no-run-id"

    def _turn_key(self, state: Any) -> str:
        messages = self._messages(state)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                not isinstance(message, HumanMessage)
                and getattr(message, "type", None) != "human"
            ):
                continue
            content = str(getattr(message, "content", ""))
            if (
                content == _CITATION_RETRY_INSTRUCTIONS
                or content == _RESEARCH_RETRY_INSTRUCTIONS
            ):
                continue
            if content.startswith(_SUMMARY_PREFIX):
                continue
            return f"{index}:{getattr(message, 'id', None)}:{message.content!r}"
        return "no-human-message"

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
