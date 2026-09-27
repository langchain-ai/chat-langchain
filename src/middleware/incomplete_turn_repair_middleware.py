"""Repair incomplete tool-only turns before a new agent invocation."""

from collections import Counter
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

_INCOMPLETE_TURN_NOTE = "This turn did not complete."


class IncompleteTurnRepairMiddleware(AgentMiddleware):
    """Close earlier turns that contain only completed tool trajectories."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Replace incomplete earlier trajectories with terminal notes."""
        messages = state.get("messages", [])
        repaired_messages = self._repair_messages(messages)
        if repaired_messages == messages:
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *repaired_messages]}

    def _repair_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        human_indexes = [
            index for index, message in enumerate(messages) if message.type == "human"
        ]
        if len(human_indexes) < 2:
            return messages

        repaired: list[AnyMessage] = []
        changed = False
        segment_start = 0
        for human_position, human_index in enumerate(human_indexes[:-1]):
            next_human_index = human_indexes[human_position + 1]
            trajectory = messages[human_index + 1 : next_human_index]
            repaired.extend(messages[segment_start : human_index + 1])
            if not self._is_incomplete_tool_trajectory(trajectory):
                repaired.extend(trajectory)
                segment_start = next_human_index
                continue
            repaired.append(AIMessage(content=_INCOMPLETE_TURN_NOTE))
            changed = True
            segment_start = next_human_index
        repaired.extend(messages[segment_start:])
        return repaired if changed else messages

    def _is_incomplete_tool_trajectory(self, messages: list[AnyMessage]) -> bool:
        tool_call_ids: list[str] = []
        tool_message_ids: list[str] = []
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_call_ids.extend(
                    tool_call["id"]
                    for tool_call in message.tool_calls
                    if tool_call.get("id")
                )
            elif isinstance(message, ToolMessage):
                tool_message_ids.append(message.tool_call_id)
            else:
                return False
        return bool(tool_call_ids) and Counter(tool_call_ids) == Counter(
            tool_message_ids
        )


__all__ = ["IncompleteTurnRepairMiddleware"]
