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
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message when it exceeds the size cap."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _truncate_content(self, content: Any) -> Any:
        """Trim user text to the cap while preserving non-text content blocks."""
        if isinstance(content, str):
            original_length = len(content)
        elif isinstance(content, list):
            original_length = sum(
                len(block) if isinstance(block, str) else len(block["text"])
                for block in content
                if isinstance(block, str)
                or (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                )
            )
        else:
            return content

        if original_length <= MAX_MESSAGE_CHARS:
            return content

        notice = (
            f"\n\n[Input truncated: original text contained {original_length} characters. "
            "Tell the user at the start of your answer that only part of the content "
            "was processed, the rest was not received, and ask them to send it in "
            "smaller parts.]"
        )
        remaining = MAX_MESSAGE_CHARS - len(notice)
        if isinstance(content, str):
            return content[:remaining] + notice

        notice_added = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                original_text = block
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                original_text = block["text"]
            else:
                truncated.append(block)
                continue

            text = original_text[:remaining]
            remaining -= len(text)
            if len(text) < len(original_text) and not notice_added:
                text += notice
                notice_added = True
            truncated.append(
                text if isinstance(block, str) else {**block, "text": text}
            )
        return truncated


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
