"""Ingress guards: input caps and credential redaction for Chat LangChain.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware only caps oversized user input and redacts credential-shaped tokens.

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
REDACTED_CREDENTIAL_PLACEHOLDER = "<REDACTED_CREDENTIAL>"

CREDENTIAL_PATTERNS = [
    re.compile(
        r"(?<![A-Za-z0-9])lsv2_(?:sk|pt)_[A-Za-z0-9]{16,}(?:_[A-Za-z0-9]+)*(?![A-Za-z0-9])"
    ),
    re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])sk-(?!ant-)[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"),
    re.compile(r"(?i)(?P<header>Authorization:\s+Bearer\s+)(?P<credential>\S+)"),
    re.compile(r"(?i)(?P<header>\bBearer\s+)(?P<credential>\S+)"),
    re.compile(r"(?i)(?P<header>\bX-Api-Key:\s*)(?P<credential>\S+)"),
]


class IngressGuardsMiddleware(AgentMiddleware):
    """Redact credentials and cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact and truncate the latest human message at agent ingress."""
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

    def _redact_secrets(self, content: Any) -> Any:
        """Replace credential-shaped tokens in user text with a placeholder."""
        if isinstance(content, str):
            return self._redact_text(content)

        if not isinstance(content, list):
            return content

        changed = False
        redacted: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted_text = self._redact_text(block)
                changed = changed or redacted_text is not block
                redacted.append(redacted_text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                redacted_text = self._redact_text(block["text"])
                changed = changed or redacted_text is not block["text"]
                redacted.append(
                    {**block, "text": redacted_text}
                    if redacted_text is not block["text"]
                    else block
                )
            else:
                redacted.append(block)
        return redacted if changed else content

    def _redact_text(self, content: str) -> str:
        redacted = content
        for pattern in CREDENTIAL_PATTERNS:
            if not pattern.search(redacted):
                continue
            if "credential" in pattern.groupindex:
                redacted = pattern.sub(
                    lambda match: match.group(0).replace(
                        match.group("credential"), REDACTED_CREDENTIAL_PLACEHOLDER
                    ),
                    redacted,
                )
            else:
                redacted = pattern.sub(REDACTED_CREDENTIAL_PLACEHOLDER, redacted)
        return redacted

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


__all__ = [
    "CREDENTIAL_PATTERNS",
    "IngressGuardsMiddleware",
    "MAX_MESSAGE_CHARS",
    "REDACTED_CREDENTIAL_PLACEHOLDER",
]
