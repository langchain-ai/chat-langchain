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
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Prune failed turns and truncate the latest user message."""
        messages = state.get("messages", [])
        update: list[Any] = self._failed_turn_removals(messages)
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    update.append(message)
                break
        return {"messages": update} if update else None

    def _failed_turn_removals(self, messages: list[Any]) -> list[RemoveMessage]:
        """Build reducer updates for incomplete turns before the current input."""
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if getattr(message, "type", None) == "human"
        ]
        if len(human_indexes) < 2:
            return []

        removals: list[RemoveMessage] = []
        for start, end in zip(human_indexes, human_indexes[1:]):
            segment = messages[start + 1 : end]
            if not segment or not all(
                self._is_failed_turn_message(message) for message in segment
            ):
                continue
            turn_messages = messages[start:end]
            if any(getattr(message, "id", None) is None for message in turn_messages):
                continue
            removals.extend(RemoveMessage(id=message.id) for message in turn_messages)
        return removals

    def _is_failed_turn_message(self, message: Any) -> bool:
        """Return whether a message belongs to a tool-only failed turn."""
        if isinstance(message, ToolMessage):
            return True
        return (
            isinstance(message, AIMessage)
            and bool(message.tool_calls)
            and not self._has_text(message.content)
        )

    def _has_text(self, content: Any) -> bool:
        """Return whether content contains non-empty textual content."""
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(self._has_text(item) for item in content)
        if isinstance(content, dict):
            return isinstance(content.get("text"), str) and bool(
                content["text"].strip()
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
