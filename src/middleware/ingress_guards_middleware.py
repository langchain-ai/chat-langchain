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
SECRET_PATTERNS = [
    re.compile(r"(?i)\b(postgres(?:ql)?|redis|mysql|mongodb)(\+\w+)?://([^:/\s]+):([^@\s]+)@"),
    re.compile(r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\blsv2_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
]
PLACEHOLDER_SECRETS = frozenset(
    {"password", "postgres", "xxxxxx", "changeme", "secret", "your_password"}
)
REDACTED_SECRET = "***REDACTED***"


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Sanitize the latest user message at agent ingress."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                redacted = self._redact_content(capped)
                if redacted is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = redacted
                    return {"messages": [message]}
                break
        return None

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

    def _redact_content(self, content: Any) -> Any:
        """Redact credential patterns while preserving non-text content blocks."""
        if isinstance(content, str):
            return self._redact_text(content)

        if not isinstance(content, list):
            return content

        changed = False
        redacted: list[Any] = []
        for block in content:
            if isinstance(block, str):
                text = self._redact_text(block)
                changed = changed or text != block
                redacted.append(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = self._redact_text(block["text"])
                changed = changed or text != block["text"]
                redacted.append({**block, "text": text})
            else:
                redacted.append(block)
        return redacted if changed else content

    def _redact_text(self, text: str) -> str:
        """Replace detected secrets in text with a stable placeholder."""
        redacted = text
        for pattern in SECRET_PATTERNS:
            if pattern is SECRET_PATTERNS[0]:
                def replace_uri(match: re.Match[str]) -> str:
                    secret = match.group(4)
                    if self._is_placeholder_secret(secret):
                        return match.group(0)
                    return f"{match.group(1)}{match.group(2) or ''}://{match.group(3)}:{REDACTED_SECRET}@"

                redacted = pattern.sub(replace_uri, redacted)
            elif pattern is SECRET_PATTERNS[1]:
                redacted = pattern.sub(
                    lambda match: re.sub(
                        r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+$",
                        rf"\1{REDACTED_SECRET}",
                        match.group(0),
                    ),
                    redacted,
                )
            else:
                redacted = pattern.sub(REDACTED_SECRET, redacted)
        return text if redacted == text else redacted

    @staticmethod
    def _is_placeholder_secret(secret: str) -> bool:
        """Return whether a credential is a documentation placeholder."""
        lowered = secret.lower()
        return lowered in PLACEHOLDER_SECRETS or lowered.startswith("your") or lowered.startswith("<")


__all__ = [
    "IngressGuardsMiddleware",
    "MAX_MESSAGE_CHARS",
    "PLACEHOLDER_SECRETS",
    "SECRET_PATTERNS",
]
