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
    re.compile(
        r"\b(?:postgres|postgresql|mysql|mongodb|mongodb\+srv|redis|rediss|amqp|clickhouse|http|https)://[^\s:@/]+:([^\s@/]{4,})@",
        re.IGNORECASE,
    ),
    re.compile(r"\b(sk-[A-Za-z0-9][A-Za-z0-9_-]{3,}|lsv2_[^\s]+|lcl_[^\s]+|ghp_[^\s]+|xoxb-[^\s]+|AIza[^\s]+|tvly-[^\s]+|pk_live_[^\s]+|AKIA[A-Z0-9]{12,})\b"),
    re.compile(r"\bBearer\s+([^\s]+)", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b"),
    re.compile(
        r"\b[\w.-]*(?:key|token|secret|password|passwd)\b\s*(?:=|:)\s*(?:[\"']?(os\.getenv\([^)]*\)|\$\{[^}]+\}|[^\"'`\s,}\]]{4,})[\"']?)",
        re.IGNORECASE,
    ),
]
ALLOWLIST_PATTERN = re.compile(
    r"(?:<REDACTED_SECRET>|YOUR_[A-Z0-9_]*|<[^>\r\n]+>|changeme|example|\$\{[^}]+\}|os\.getenv\([^)]*\))",
    re.IGNORECASE,
)
REDACTED_SECRET = "<REDACTED_SECRET>"


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Cap and redact the latest user message at agent ingress."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                redacted = self._redact_secrets(capped)
                if redacted is not message.content:
                    message.content = redacted
                    return {"messages": [message]}
                break
        return None

    def _redact_secrets(self, content: Any) -> Any:
        """Replace detected secret values while preserving surrounding content."""
        if isinstance(content, str):
            redacted = content
            for pattern in SECRET_PATTERNS:
                redacted = pattern.sub(self._redact_match, redacted)
            return redacted if redacted != content else content

        if not isinstance(content, list):
            return content

        changed = False
        redacted_content: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted = self._redact_secrets(block)
                changed = changed or redacted is not block
                redacted_content.append(redacted)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                redacted = self._redact_secrets(block["text"])
                changed = changed or redacted is not block["text"]
                redacted_content.append({**block, "text": redacted})
            else:
                redacted_content.append(block)
        return redacted_content if changed else content

    @staticmethod
    def _redact_match(match: re.Match[str]) -> str:
        candidate = match.group(1)
        if ALLOWLIST_PATTERN.fullmatch(candidate):
            return match.group(0)
        start, end = match.span(1)
        return f"{match.group(0)[: start - match.start()]}{REDACTED_SECRET}{match.group(0)[end - match.start() :]}"

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
    "ALLOWLIST_PATTERN",
    "IngressGuardsMiddleware",
    "MAX_MESSAGE_CHARS",
    "REDACTED_SECRET",
    "SECRET_PATTERNS",
]
