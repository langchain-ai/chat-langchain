"""Ingress guards: input caps for Chat LangChain on Managed Deep Agents.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware only caps oversized user input.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py`` — nested ``before_agent`` spans cannot reliably update the
LangSmith root run. Redaction counts are therefore attached to this middleware
span. Git-linked host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are
not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

from typing import Any

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from src.utils.secret_redaction import redact_secrets

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact secrets in human messages and truncate the latest one."""
        messages = state.get("messages", [])
        changed_messages: list[Any] = []
        redaction_count = 0
        latest_human = next(
            (
                message
                for message in reversed(messages)
                if getattr(message, "type", None) == "human"
            ),
            None,
        )
        for message in messages:
            if getattr(message, "type", None) == "human":
                redacted, count = self._redact_content(message.content)
                redaction_count += count
                capped = (
                    self._truncate_content(redacted)
                    if message is latest_human
                    else redacted
                )
                if capped is not message.content:
                    message.content = capped
                    changed_messages.append(message)
        if redaction_count:
            self._track_redaction_metadata(redaction_count)
        return {"messages": changed_messages} if changed_messages else None

    def _redact_content(self, content: Any) -> tuple[Any, int]:
        """Redact secrets in text while preserving non-text content blocks."""
        if isinstance(content, str):
            return redact_secrets(content)
        if not isinstance(content, list):
            return content, 0

        redacted_blocks: list[Any] = []
        count = 0
        changed = False
        for block in content:
            if isinstance(block, str):
                redacted, block_count = redact_secrets(block)
                redacted_blocks.append(redacted)
                count += block_count
                changed = changed or redacted != block
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                redacted, block_count = redact_secrets(block["text"])
                redacted_blocks.append({**block, "text": redacted})
                count += block_count
                changed = changed or redacted != block["text"]
            else:
                redacted_blocks.append(block)
        return (redacted_blocks if changed else content), count

    def _track_redaction_metadata(self, count: int) -> None:
        """Record redactions on the middleware span for trace visibility."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree:
                run_tree.metadata["secrets_redacted"] = count
        except Exception:
            pass

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
