"""Repair checkpointed turns that ended without an assistant answer."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

ABORTED_TURN_MESSAGE = "(the previous turn did not complete and returned no answer)"


class AbortedTurnCleanupMiddleware(AgentMiddleware):
    """Repair historical turns that contain no assistant text answer."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Replace incomplete historical turns with short assistant placeholders."""
        messages = list(state.get("messages", []))
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]
        if len(human_indexes) < 2:
            return None

        repaired: list[AnyMessage] = []
        changed = False
        for start, end in zip(human_indexes, human_indexes[1:]):
            segment = messages[start:end]
            if any(self._has_non_empty_text(message) for message in segment):
                repaired.extend(segment)
                continue

            tool_call_ids = {
                tool_call.get("id")
                for message in segment
                if isinstance(message, AIMessage)
                for tool_call in message.tool_calls
                if tool_call.get("id")
            }
            segment_without_orphans = [
                message
                for message in segment[1:]
                if not (
                    isinstance(message, AIMessage)
                    and message.tool_calls
                )
                and not (
                    isinstance(message, ToolMessage)
                    and message.tool_call_id in tool_call_ids
                )
            ]
            repaired.extend(
                [
                    messages[start],
                    AIMessage(content=ABORTED_TURN_MESSAGE),
                    *segment_without_orphans,
                ]
            )
            changed = True

        repaired.extend(messages[human_indexes[-1] :])

        if not changed:
            return None

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *repaired,
            ]
        }

    def wrap_model_call(self, request, handler):
        """Turn model failures into a persisted terminating assistant message."""
        try:
            return handler(request)
        except Exception:
            return AIMessage(content=ABORTED_TURN_MESSAGE)

    async def awrap_model_call(self, request, handler):
        """Turn async model failures into a terminating assistant message."""
        try:
            return await handler(request)
        except Exception:
            return AIMessage(content=ABORTED_TURN_MESSAGE)

    @staticmethod
    def _has_non_empty_text(message: AnyMessage) -> bool:
        if not isinstance(message, AIMessage):
            return False

        content = message.content
        if isinstance(content, str):
            return bool(content.strip())
        if not isinstance(content, list):
            return False

        for block in content:
            if isinstance(block, str) and block.strip():
                return True
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].strip()
            ):
                return True
        return False


__all__ = ["ABORTED_TURN_MESSAGE", "AbortedTurnCleanupMiddleware"]
