"""Ingress guards: input caps for Chat LangChain on Managed Deep Agents.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware only caps oversized user input.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py`` — nested ``before_agent`` spans cannot reliably update the
LangSmith root run. Git-linked host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are
not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Repair abandoned turns and truncate the latest user message."""
        messages = state.get("messages", [])
        updates = self._remove_incomplete_turns(messages)
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    updates.append(message)
                break
        return {"messages": updates} if updates else None

    def _remove_incomplete_turns(self, messages: list[Any]) -> list[Any]:
        """Remove earlier human turns that never received a text answer."""
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if getattr(message, "type", None) == "human"
        ]
        if len(human_indexes) < 2:
            return []

        remove_indexes: set[int] = set()
        for position, start in enumerate(human_indexes[:-1]):
            end = human_indexes[position + 1]
            turn_messages = messages[start:end]
            has_text_answer = any(
                isinstance(message, AIMessage)
                and self._has_text_content(message.content)
                for message in turn_messages
            )
            if not has_text_answer:
                remove_indexes.update(range(start, end))

        if not remove_indexes:
            return []

        return [
            RemoveMessage(id=messages[index].id)
            for index in sorted(remove_indexes)
            if messages[index].id is not None
        ]

    def _has_text_content(self, content: Any) -> bool:
        """Return whether content contains non-empty text."""
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(
                self._has_text_content(block)
                if isinstance(block, str)
                else isinstance(block, dict)
                and block.get("type") == "text"
                and self._has_text_content(block.get("text"))
                for block in content
            )
        return False

    def _truncate_content(self, content: Any) -> Any:
        """Trim user text to the cap while preserving non-text content blocks."""
        if isinstance(content, str):
            return (
                content[:MAX_MESSAGE_CHARS]
                if len(content) > MAX_MESSAGE_CHARS
                else content
            )

        if not isinstance(content, list):
            return content

        remaining = MAX_MESSAGE_CHARS
        changed = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                text = block[:remaining]
                changed = changed or len(text) != len(block)
                truncated.append(text)
                remaining -= len(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = block["text"][:remaining]
                changed = changed or len(text) != len(block["text"])
                truncated.append({**block, "text": text})
                remaining -= len(text)
            else:
                truncated.append(block)
        return truncated if changed else content


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
