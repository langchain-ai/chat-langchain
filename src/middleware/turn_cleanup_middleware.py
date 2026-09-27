"""Clean up persisted state left by an aborted agent turn."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

FAILED_TURN_MESSAGE = "I couldn't complete that turn. Please try again."


def _has_completed_response(messages: list[Any]) -> bool:
    return any(
        isinstance(message, AIMessage)
        and bool(message.content)
        and not message.tool_calls
        for message in messages
    )


def _cleanup_aborted_turn(messages: list[Any]) -> tuple[list[Any], bool]:
    cleaned: list[Any] = []
    changed = False
    index = 0

    while index < len(messages):
        message = messages[index]
        if not isinstance(message, HumanMessage):
            cleaned.append(message)
            index += 1
            continue

        next_human = index + 1
        while next_human < len(messages) and not isinstance(
            messages[next_human], HumanMessage
        ):
            next_human += 1
        turn_messages = messages[index + 1 : next_human]
        has_partial_output = any(
            isinstance(turn_message, (AIMessage, ToolMessage))
            for turn_message in turn_messages
        )
        if has_partial_output and not _has_completed_response(turn_messages):
            cleaned.extend([message, AIMessage(content=FAILED_TURN_MESSAGE)])
            changed = True
        else:
            cleaned.extend(messages[index:next_human])
        index = next_human

    return cleaned, changed


class TurnCleanupMiddleware(AgentMiddleware):
    """Close aborted turns before processing the next user message."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Persist cleanup for any aborted prior turn."""
        messages = list(state.get("messages", []))
        cleaned, changed = _cleanup_aborted_turn(messages)
        if not changed:
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *cleaned]}


__all__ = ["FAILED_TURN_MESSAGE", "TurnCleanupMiddleware"]
