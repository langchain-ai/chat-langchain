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

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000

_REDACTION_PATTERNS = [
    re.compile(
        r"((?:postgres|postgresql|mysql|mongodb|redis|amqp|https?)(?:\+\w+)?://[^:/\s@]+:)([^@\s]{6,})(@)",
        re.I,
    ),
    re.compile(
        r"((?:sk-ant-|sk-|tvly-|AIza|ghp_|xoxb-|pk_live_|lsv2_|lcl_)[^\s\"'`,;)]{6,})",
        re.I,
    ),
]


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact secrets and truncate the latest user message."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                redacted = self._redact_content(message.content)
                capped = self._truncate_content(redacted)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _redact_content(self, content: Any) -> Any:
        """Replace detected secrets while preserving non-text content blocks."""
        if isinstance(content, str):
            redacted = content
            for pattern in _REDACTION_PATTERNS:
                if pattern is _REDACTION_PATTERNS[0]:
                    redacted = pattern.sub(self._redact_uri_match, redacted)
                else:
                    redacted = pattern.sub(self._redact_token_match, redacted)
            return redacted if redacted != content else content

        if not isinstance(content, list):
            return content

        changed = False
        redacted_content: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted = self._redact_content(block)
                changed = changed or redacted != block
                redacted_content.append(redacted)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = self._redact_content(block["text"])
                changed = changed or text != block["text"]
                redacted_content.append({**block, "text": text})
            else:
                redacted_content.append(block)
        return redacted_content if changed else content

    @staticmethod
    def _redact_uri_match(match: re.Match[str]) -> str:
        secret = match.group(2)
        return match.group(0) if _is_placeholder(secret) else f"{match.group(1)}<REDACTED_SECRET>{match.group(3)}"

    @staticmethod
    def _redact_token_match(match: re.Match[str]) -> str:
        secret = match.group(1)
        return match.group(0) if _is_placeholder(secret) else "<REDACTED_SECRET>"

    def _truncate_content(self, content: Any) -> Any:
        """Trim user text to the cap while preserving non-text content blocks."""
        if isinstance(content, str):
            return content[:MAX_MESSAGE_CHARS] if len(content) > MAX_MESSAGE_CHARS else content

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


def _is_placeholder(value: str) -> bool:
    """Return whether a value resembles a placeholder or environment lookup."""
    normalized = value.lower()
    return (
        normalized.startswith("your")
        or (value.startswith("<") and value.endswith(">"))
        or normalized == "xxx"
        or "os.getenv" in normalized
        or value.startswith("${")
    )
