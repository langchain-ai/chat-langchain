"""Ingress guards for Chat LangChain on Managed Deep Agents.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware caps oversized user input and reconciles unanswered stopped turns.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py`` — nested ``before_agent`` spans cannot reliably update the
LangSmith root run. Git-linked host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are
not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import RemoveMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap user input and reconcile stopped turns at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Cap the latest user message and merge unanswered turns as context."""
        messages = state.get("messages", [])
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                stale_messages = []
                for earlier in reversed(messages[:index]):
                    if getattr(earlier, "type", None) == "ai":
                        break
                    if getattr(earlier, "type", None) == "human":
                        stale_messages.append(earlier)
                stale_messages.reverse()
                latest_text = self._text_content(message.content)
                stale_texts = [
                    self._text_content(earlier.content)[:500]
                    for earlier in stale_messages
                    if self._text_content(earlier.content) != latest_text
                ]
                if stale_texts:
                    prefix = (
                        "[Stopped-turn context]\n"
                        + "\n".join(f"- {text}" for text in stale_texts)
                        + "\n[/Stopped-turn context]\n"
                        "Answer the latest message; earlier messages were not answered "
                        "because the previous run was stopped - address them only if "
                        "the latest message refers to them.\n\n"
                    )
                    capped = (
                        prefix + capped
                        if isinstance(capped, str)
                        else [{"type": "text", "text": prefix}, *capped]
                    )
                if stale_messages or capped is not message.content:
                    return {
                        "messages": [
                            *[
                                RemoveMessage(id=cast(str, earlier.id))
                                for earlier in stale_messages
                            ],
                            message.model_copy(update={"content": capped}),
                        ]
                    }
                return None
        return None

    def _text_content(self, content: Any) -> str:
        """Extract text while ignoring non-text content blocks."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        return "".join(
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str)
            or (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
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
