"""Ingress guards: input caps and stopped-turn context for Chat LangChain.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware caps oversized user input and marks unanswered earlier turns as context.

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
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000
STOPPED_MESSAGE_PREFIX = (
    "[Earlier message sent before the previous answer was stopped - "
    "context only, not the current question]\n"
)


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap user input and distinguish stopped turns from the current question."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Cap the latest user message and mark trailing earlier ones as context."""
        messages = state.get("messages", [])
        updated_messages = []
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    updated_messages.append(
                        message.model_copy(update={"content": capped})
                    )
                break

        trailing_humans = []
        for message in reversed(messages):
            if getattr(message, "type", None) != "human":
                break
            trailing_humans.append(message)
        for message in reversed(trailing_humans[1:]):
            marked = self._mark_stopped_content(message.content)
            if marked is not message.content:
                updated_messages.append(message.model_copy(update={"content": marked}))

        return {"messages": updated_messages} if updated_messages else None

    def _mark_stopped_content(self, content: Any) -> Any:
        """Prefix stopped-turn content once while preserving content blocks."""
        if isinstance(content, str):
            return (
                content
                if content.startswith(STOPPED_MESSAGE_PREFIX)
                else STOPPED_MESSAGE_PREFIX + content
            )
        if not isinstance(content, list):
            return content
        if content:
            first_block = content[0]
            first_text = (
                first_block
                if isinstance(first_block, str)
                else (
                    first_block.get("text")
                    if isinstance(first_block, dict)
                    and first_block.get("type") == "text"
                    else None
                )
            )
            if isinstance(first_text, str) and first_text.startswith(
                STOPPED_MESSAGE_PREFIX
            ):
                return content
        return [{"type": "text", "text": STOPPED_MESSAGE_PREFIX}, *content]

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
