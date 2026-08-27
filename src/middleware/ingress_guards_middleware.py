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

REDACTION_RULES = [
    re.compile(
        r"(?i)\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|https?)(\+\w+)?://([^\s:/@]+):(?P<secret>[^\s@/]{6,})@"
    ),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)(?P<secret>[A-Za-z0-9._\-]{16,})"),
    re.compile(r"(?i)(sk-|lsv2_|ghp_|AKIA)(?P<secret>[A-Za-z0-9._\-]{16,})"),
]

_PLACEHOLDER_VALUES = {
    "password",
    "pass",
    "postgres",
    "changeme",
    "your_password",
    "your-langsmith-api-key",
}


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact secrets and truncate the latest user message."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                redacted = self._redact_secrets(message.content)
                capped = self._truncate_content(redacted)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

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

    def _redact_secrets(self, content: Any) -> Any:
        """Redact recognized secrets while preserving non-text content blocks."""
        if isinstance(content, str):
            redacted = self._redact_text(content)
            return redacted if redacted != content else content

        if not isinstance(content, list):
            return content

        changed = False
        redacted_content: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted_block = self._redact_text(block)
                changed = changed or redacted_block != block
                redacted_content.append(redacted_block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = block["text"]
                redacted_text = self._redact_text(text)
                changed = changed or redacted_text != text
                redacted_content.append({**block, "text": redacted_text})
            else:
                redacted_content.append(block)
        return redacted_content if changed else content

    def _redact_text(self, content: str) -> str:
        """Apply each redaction rule to text content."""
        redacted = content
        for pattern in REDACTION_RULES:
            redacted = pattern.sub(self._replace_secret, redacted)
        return redacted

    @staticmethod
    def _replace_secret(match: re.Match[str]) -> str:
        """Replace a matched secret unless it is a documentation placeholder."""
        secret = match.group("secret")
        if IngressGuardsMiddleware._is_placeholder(secret):
            return match.group(0)
        start, end = match.span("secret")
        return (
            match.group(0)[: start - match.start()]
            + "[REDACTED]"
            + match.group(0)[end - match.start() :]
        )

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        """Return whether a value is an allowed documentation stand-in."""
        normalized = value.lower()
        return (
            normalized in _PLACEHOLDER_VALUES
            or (value.startswith("<") and value.endswith(">"))
            or (value.startswith("${") and value.endswith("}"))
            or len(set(value)) < 5
        )


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS", "REDACTION_RULES"]
