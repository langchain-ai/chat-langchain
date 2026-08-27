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
PLACEHOLDER_VALUES = frozenset(
    {
        "password",
        "passwd",
        "pass",
        "secret",
        "token",
        "username",
        "user",
        "admin",
        "postgres",
        "changeme",
        "redacted",
        "langchain",
        "your_password",
    }
)
CREDENTIAL_PATTERNS = [
    re.compile(
        r"(?P<scheme>[a-z][a-z0-9+.\-]*)://(?P<user>[^\s:/@]+):(?P<pw>[^\s@]{6,})@"
    ),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"lsv2_(pt|sk)_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"ey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
    ),
    re.compile(
        r"(?i)(password|passwd|api[_ -]?key|secret|token)\s*[:=]\s*[\"']?(?P<val>[^\s\"']{8,})[\"']?"
    ),
]
_PLACEHOLDER_PATTERN = re.compile(
    r"(?i)your[-_]?|^<|^\{|xxx|placeholder|changeme|os\.getenv|getenv\(|process\.env"
)


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered in PLACEHOLDER_VALUES
        or _PLACEHOLDER_PATTERN.search(value) is not None
        or len(set(value)) < 6
        or not (re.search(r"[A-Za-z]", value) and re.search(r"[^A-Za-z]", value))
    )


def redact_secrets(text: str) -> tuple[str, int]:
    """Redact credential-shaped values from user-provided text."""
    redactions = 0
    redacted = text
    for pattern in CREDENTIAL_PATTERNS:
        group_name = "pw" if "pw" in pattern.groupindex else "val"

        def replace(match: re.Match[str]) -> str:
            nonlocal redactions
            candidate = match.group(group_name) if group_name in match.groupdict() else match.group(0)
            if _is_placeholder(candidate):
                return match.group(0)
            redactions += 1
            start, end = match.span(group_name) if group_name in match.groupdict() else match.span()
            return match.group(0)[: start - match.start()] + "[REDACTED_CREDENTIAL]" + match.group(0)[end - match.start() :]

        redacted = pattern.sub(replace, redacted)
    return redacted, redactions


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact credentials and truncate the latest user message."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                content = message.content
                redacted, redaction_count = self._redact_content(content)
                capped = self._truncate_content(redacted)
                if redaction_count or capped is not content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _redact_content(self, content: Any) -> tuple[Any, int]:
        if isinstance(content, str):
            return redact_secrets(content)

        if not isinstance(content, list):
            return content, 0

        redactions = 0
        redacted: list[Any] = []
        for block in content:
            if isinstance(block, str):
                block_text, block_redactions = redact_secrets(block)
                redacted.append(block_text)
                redactions += block_redactions
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                block_text, block_redactions = redact_secrets(block["text"])
                redacted.append({**block, "text": block_text})
                redactions += block_redactions
            else:
                redacted.append(block)
        return (redacted, redactions) if redactions else (content, 0)

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
    "CREDENTIAL_PATTERNS",
    "IngressGuardsMiddleware",
    "MAX_MESSAGE_CHARS",
    "PLACEHOLDER_VALUES",
    "redact_secrets",
]
