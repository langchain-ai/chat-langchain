"""Remove abandoned user turns before the agent runs."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import RemoveMessage
from langgraph.runtime import Runtime


class DanglingTurnMiddleware(AgentMiddleware):
    """Remove human turns that have no prose answer before a later question."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Remove abandoned turns while preserving the latest human message."""
        messages = state.get("messages", [])
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if getattr(message, "type", None) == "human"
        ]
        removals: list[RemoveMessage] = []

        for start, end in zip(human_indexes, human_indexes[1:]):
            block = messages[start:end]
            if all(self._is_abandoned_message(message) for message in block[1:]):
                removals.extend(RemoveMessage(id=message.id) for message in block)

        return {"messages": removals} if removals else None

    def _is_abandoned_message(self, message: Any) -> bool:
        """Check whether a message can belong to an unanswered turn."""
        message_type = getattr(message, "type", None)
        if message_type == "tool":
            return True
        if message_type != "ai":
            return False
        return not getattr(message, "text", "").strip()


__all__ = ["DanglingTurnMiddleware"]
