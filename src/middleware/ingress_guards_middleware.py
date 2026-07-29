"""Ingress guards: secret redaction and input caps for Chat LangChain on MDA.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware redacts credential literals from and caps oversized user input.

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

#: Token substituted for any credential-shaped literal found in user input.
REDACTION_TOKEN = "<REDACTED_CREDENTIAL>"

#: Credential-shaped literals that must never reach the model context.
CREDENTIAL_PATTERN = re.compile(
    r"sk-ant-[A-Za-z0-9\-_]{20,}"
    r"|sk-[A-Za-z0-9]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|AIza[0-9A-Za-z\-_]{25,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|lsv2_(?:pt|sk)_[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
)

#: Substrings marking an obvious placeholder that should be left alone.
_PLACEHOLDER_HINTS = ("xxx", "...", "your", "replace", "abc123", "1234567890")


class IngressGuardsMiddleware(AgentMiddleware):
    """Redact credentials in, and cap the size of, user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact secrets in, then truncate, the latest user message."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(self._redact_secrets(message.content))
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _redact_secrets(self, content: Any) -> Any:
        """Replace credential-shaped literals with ``REDACTION_TOKEN``."""
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
        """Substitute every non-placeholder credential match in ``text``."""

        def _sub(match: re.Match[str]) -> str:
            value = match.group(0)
            lowered = value.lower()
            if any(hint in lowered for hint in _PLACEHOLDER_HINTS):
                return value
            return REDACTION_TOKEN

        redacted = CREDENTIAL_PATTERN.sub(_sub, text)
        # Preserve object identity so ``before_agent`` still no-ops on clean input.
        return text if redacted == text else redacted

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


__all__ = [
    "CREDENTIAL_PATTERN",
    "IngressGuardsMiddleware",
    "MAX_MESSAGE_CHARS",
    "REDACTION_TOKEN",
]
