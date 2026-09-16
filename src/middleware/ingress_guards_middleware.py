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
REDACTION_PLACEHOLDER = "<REDACTED_CREDENTIAL>"
_HEADER_PATTERN = re.compile(
    r"""(?i)(\b(?:x-api-key|authorization)\s*:\s*(?:(?:Bearer)\s+)?)(["']?)([^\s"']{20,})(\2)"""
)
CREDENTIAL_PATTERNS = (
    re.compile(r"lsv2_(?:sk|pt)_[A-Za-z0-9]{20,}(?:_[A-Za-z0-9]+)*(?![A-Za-z0-9_])"),
    re.compile(r"ls__[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}(?![A-Za-z0-9])"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}(?![A-Za-z0-9_])"),
    re.compile(r"AKIA[0-9A-Z]{16}(?![0-9A-Z])"),
    re.compile(
        r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?![A-Za-z0-9_-])"
    ),
    _HEADER_PATTERN,
)


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact credentials and truncate the latest user message."""
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

    def _redact_text(self, text: str) -> str:
        """Replace credential-shaped values in text."""
        redacted = text
        for pattern in CREDENTIAL_PATTERNS:
            if pattern is _HEADER_PATTERN:
                redacted = pattern.sub(
                    lambda match: (
                        f"{match.group(1)}{match.group(2)}"
                        f"{REDACTION_PLACEHOLDER}{match.group(4)}"
                    ),
                    redacted,
                )
            else:
                redacted = pattern.sub(REDACTION_PLACEHOLDER, redacted)
        return redacted

    def _redact_content(self, content: Any) -> Any:
        """Redact credentials in text content while preserving other blocks."""
        if isinstance(content, str):
            redacted = self._redact_text(content)
            return redacted if redacted != content else content

        if not isinstance(content, list):
            return content

        changed = False
        redacted_content: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted = self._redact_text(block)
                changed = changed or redacted != block
                redacted_content.append(redacted)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = block["text"]
                redacted = self._redact_text(text)
                changed = changed or redacted != text
                redacted_content.append({**block, "text": redacted})
            else:
                redacted_content.append(block)
        return redacted_content if changed else content

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
