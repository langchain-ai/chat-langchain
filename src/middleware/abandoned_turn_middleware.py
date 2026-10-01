"""Clean up interrupted docs agent turns before the next model call."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

INTERRUPTED_TURN_MARKER = (
    "The previous request was interrupted before an answer was completed."
)


class AbandonedTurnMiddleware(AgentMiddleware[AgentState]):
    """Replace abandoned tool-call chains with interruption markers."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Replace abandoned turns with interruption markers."""
        messages = list(state.get("messages", []))
        human_indices = [
            index
            for index, message in enumerate(messages)
            if getattr(message, "type", None) == "human"
        ]
        if len(human_indices) < 2:
            return None

        updates: list[RemoveMessage | AIMessage] = []
        changed = False
        for human_position, human_index in enumerate(human_indices[:-1]):
            segment_end = human_indices[human_position + 1]
            segment = messages[human_index + 1 : segment_end]
            if any(
                isinstance(message, AIMessage)
                and self._has_textual_content(message.content)
                for message in segment
            ):
                continue

            tool_only_messages = [
                message
                for message in segment
                if isinstance(message, AIMessage) and self._is_tool_only(message)
            ]
            tool_call_ids = {
                call.get("id")
                for message in tool_only_messages
                for call in message.tool_calls
                if call.get("id")
            }
            removable_ids = {
                message.id for message in tool_only_messages if message.id is not None
            }
            removable_ids.update(
                message.id
                for message in segment
                if isinstance(message, ToolMessage)
                and message.tool_call_id in tool_call_ids
                and message.id is not None
            )
            marker_target = next(
                (
                    message
                    for message in segment
                    if message.id is not None and message.id in removable_ids
                ),
                None,
            )
            if marker_target is None:
                continue

            updates.extend(RemoveMessage(id=message_id) for message_id in removable_ids)
            updates.append(
                AIMessage(content=INTERRUPTED_TURN_MARKER, id=marker_target.id)
            )
            changed = True

        return {"messages": updates} if changed else None

    @staticmethod
    def _is_tool_only(message: AIMessage) -> bool:
        return bool(
            message.tool_calls
        ) and not AbandonedTurnMiddleware._has_textual_content(message.content)

    @staticmethod
    def _has_textual_content(content: Any) -> bool:
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(
                isinstance(block, dict)
                and isinstance(block.get("text"), str)
                and block["text"].strip()
                for block in content
            )
        return False


__all__ = ["AbandonedTurnMiddleware", "INTERRUPTED_TURN_MARKER"]
