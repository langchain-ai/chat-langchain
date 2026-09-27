"""Remove unanswered human turns left by failed agent runs."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import RemoveMessage
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class OrphanedTurnMiddleware(AgentMiddleware):
    """Strip human turns that have no content-bearing assistant response."""

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Remove orphaned turns before the next model call."""
        messages = state.get("messages", [])
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if getattr(message, "type", None) == "human"
        ]
        orphaned_indexes: set[int] = set()
        orphaned_turns = 0

        for human_index, next_human_index in zip(human_indexes, human_indexes[1:]):
            has_assistant_content = any(
                self._has_content(message)
                for message in messages[human_index + 1 : next_human_index]
                if getattr(message, "type", None) == "ai"
            )
            if not has_assistant_content:
                orphaned_indexes.update(range(human_index, next_human_index))
                orphaned_turns += 1

        if not orphaned_indexes:
            return None

        retained_messages = [
            message
            for index, message in enumerate(messages)
            if index not in orphaned_indexes
        ]
        self._record_cleanup(orphaned_turns)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *retained_messages,
            ]
        }

    @staticmethod
    def _has_content(message: Any) -> bool:
        content = getattr(message, "content", None)
        return bool(content)

    @staticmethod
    def _record_cleanup(orphaned_turns: int) -> None:
        try:
            callbacks = get_config().get("callbacks")
        except RuntimeError:
            return
        if callbacks is not None:
            callbacks.add_metadata(
                {
                    "orphaned_turn_stripped": True,
                    "orphaned_turn_count": orphaned_turns,
                }
            )


__all__ = ["OrphanedTurnMiddleware"]
