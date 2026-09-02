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
SECRET_PLACEHOLDER = "YOUR_API_KEY_HERE"
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"lsv2_(?:sk|pt|ak)_[A-Za-z0-9_]{16,}"),
    re.compile(r"lcl_[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{25,}"),
    re.compile(r"xoxb-[0-9A-Za-z-]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9]{16,}"),
    re.compile(r"pk_live_[A-Za-z0-9]{16,}"),
    re.compile(r"sk_live_[A-Za-z0-9]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(
        r"(?i)(x-api-key|authorization|api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?(?:Bearer\s+)?([A-Za-z0-9_\-.]{16,})"
    ),
]


def _redact_secrets(content: Any) -> Any:
    """Replace credential-like values with a stable placeholder."""
    if isinstance(content, str):
        redacted = content
        for pattern in SECRET_PATTERNS:
            if pattern.groups:
                redacted = pattern.sub(
                    lambda match: (
                        match.group(0)[: match.start(2) - match.start(0)]
                        + SECRET_PLACEHOLDER
                        + match.group(0)[match.end(2) - match.start(0) :]
                    ),
                    redacted,
                )
            else:
                redacted = pattern.sub(SECRET_PLACEHOLDER, redacted)
        return redacted if redacted != content else content

    if not isinstance(content, list):
        return content

    changed = False
    redacted_content: list[Any] = []
    for block in content:
        if isinstance(block, str):
            redacted_block = _redact_secrets(block)
            changed = changed or redacted_block is not block
            redacted_content.append(redacted_block)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            redacted_text = _redact_secrets(block["text"])
            changed = changed or redacted_text is not block["text"]
            redacted_content.append({**block, "text": redacted_text})
        else:
            redacted_content.append(block)
    return redacted_content if changed else content


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message when it exceeds the size cap."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                redacted = _redact_secrets(message.content)
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


__all__ = [
    "IngressGuardsMiddleware",
    "MAX_MESSAGE_CHARS",
    "SECRET_PATTERNS",
    "SECRET_PLACEHOLDER",
    "_redact_secrets",
]
