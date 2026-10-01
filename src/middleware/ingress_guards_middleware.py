"""Ingress guards: input caps for Chat LangChain on Managed Deep Agents.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware cleans up incomplete turns and caps oversized user input.

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
from langchain_core.messages import RemoveMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Clean up incomplete turns and truncate the latest user message."""
        messages = state.get("messages", [])
        human_indexes = [
            index
            for index, message in enumerate(messages)
            if getattr(message, "type", None) == "human"
        ]
        updates: list[Any] = [
            RemoveMessage(id=message.id)
            for start, end in zip(human_indexes, human_indexes[1:])
            if not self._turn_completed(messages[start + 1 : end])
            for message in messages[start:end]
        ]

        if human_indexes:
            latest_human = messages[human_indexes[-1]]
            capped = self._truncate_content(latest_human.content)
            if capped is not latest_human.content:
                # Same id => the messages reducer overwrites in place.
                latest_human.content = capped
                updates.append(latest_human)

        return {"messages": updates} if updates else None

    def _turn_completed(self, messages: list[Any]) -> bool:
        return any(
            getattr(message, "type", None) == "ai"
            and not getattr(message, "tool_calls", [])
            and self._has_text_content(getattr(message, "content", None))
            for message in messages
        )

    def _has_text_content(self, content: Any) -> bool:
        if isinstance(content, str):
            return bool(content.strip())
        if not isinstance(content, list):
            return False
        return any(
            isinstance(block, str)
            and block.strip()
            or isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block["text"].strip()
            for block in content
        )

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
