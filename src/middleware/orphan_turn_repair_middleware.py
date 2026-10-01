"""Repair answerless checkpointed turns before agent execution."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.runtime import Runtime

_PLACEHOLDER = "The previous turn failed before producing an answer."


class OrphanTurnRepairMiddleware(AgentMiddleware):
    """Repair answerless turns before the next agent invocation."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Replace orphaned turns with placeholders in message order."""
        messages = list(state.get("messages", []))
        turns = self._turns(messages)
        orphaned_turns = {
            index
            for index, (start, end) in enumerate(turns[:-1])
            if not self._has_textual_ai(messages[start + 1 : end])
        }
        if not orphaned_turns:
            return None

        first_orphan_start = turns[min(orphaned_turns)][0]
        suffix = messages[first_orphan_start:]
        replacements: list[BaseMessage] = []
        for index, (start, end) in enumerate(turns):
            if end <= first_orphan_start:
                continue
            if start < first_orphan_start:
                continue
            relative_start = start - first_orphan_start
            relative_end = end - first_orphan_start
            if index in orphaned_turns:
                replacements.append(self._fresh_message(messages[start]))
                replacements.append(AIMessage(content=_PLACEHOLDER))
            else:
                replacements.extend(
                    self._fresh_message(message)
                    for message in suffix[relative_start:relative_end]
                )

        removals = [RemoveMessage(id=message.id) for message in suffix]
        return {"messages": [*removals, *replacements]}

    def _turns(self, messages: list[BaseMessage]) -> list[tuple[int, int]]:
        starts = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]
        return [
            (start, starts[index + 1] if index + 1 < len(starts) else len(messages))
            for index, start in enumerate(starts)
        ]

    def _has_textual_ai(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and self._has_text(message.content)
            for message in messages
        )

    def _has_text(self, content: Any) -> bool:
        if isinstance(content, str):
            return bool(content.strip())
        if not isinstance(content, list):
            return False
        return any(
            (isinstance(block, str) and bool(block.strip()))
            or (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and bool(block["text"].strip())
            )
            for block in content
        )

    def _fresh_message(self, message: BaseMessage) -> BaseMessage:
        return message.model_copy(update={"id": None})


__all__ = ["OrphanTurnRepairMiddleware"]
