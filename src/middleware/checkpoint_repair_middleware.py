"""Repair incomplete message turns before the agent runs."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

NO_AI_RESPONSE_MESSAGE = (
    "I couldn't complete that request because the agent reached its execution "
    "limit. Please try again."
)


class CheckpointRepairMiddleware(AgentMiddleware[AgentState]):
    """Remove unanswered turns and provide a fallback for empty results."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Remove incomplete earlier turns before processing the current question."""
        messages = state.get("messages", [])
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]
        if len(human_indexes) < 2:
            return None

        removed_ids: list[str] = []
        for human_position, human_index in enumerate(human_indexes[:-1]):
            next_human_index = human_indexes[human_position + 1]
            turn_messages = messages[human_index + 1 : next_human_index]
            if any(
                isinstance(message, AIMessage) and self._has_text(message.content)
                for message in turn_messages
            ):
                continue

            turn_ids = [
                message.id
                for message in messages[human_index:next_human_index]
                if message.id is not None
                and (
                    isinstance(message, HumanMessage)
                    or isinstance(message, ToolMessage)
                    or (
                        isinstance(message, AIMessage)
                        and not self._has_text(message.content)
                    )
                )
            ]
            removed_ids.extend(turn_ids)

        if not removed_ids:
            return None

        runtime.stream_writer({"checkpoint_repair_removed_messages": len(removed_ids)})
        return {
            "messages": [RemoveMessage(id=message_id) for message_id in removed_ids]
        }

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Return a retryable error when the graph produced no AI text."""
        if any(
            isinstance(message, AIMessage) and self._has_text(message.content)
            for message in state.get("messages", [])
        ):
            return None
        return {"messages": [AIMessage(content=NO_AI_RESPONSE_MESSAGE)]}

    def _has_text(self, content: Any) -> bool:
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(self._has_text(block) for block in content)
        if isinstance(content, dict):
            if content.get("type") == "text":
                return self._has_text(content.get("text"))
            return False
        return False


__all__ = ["CheckpointRepairMiddleware", "NO_AI_RESPONSE_MESSAGE"]
