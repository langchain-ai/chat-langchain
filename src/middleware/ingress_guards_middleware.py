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
        r"(?P<scheme>[a-z][a-z0-9+.-]*://[^:/\s@]+:)(?P<secret>[^@\s/]{4,})@",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|POSTGRES_URI\w*|DATABASE_URL|REDIS_URI)"
        r"\s*[=:]\s*(?P<secret>(?![a-z][a-z0-9+.-]*://)\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<secret>(?:sk-|ghp_|lsv2_pt_|lsv2_sk_)[A-Za-z0-9_\-]+|AKIA[0-9A-Z]{16})"
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
                redacted = self._redact_secrets(message.content)
                capped = self._truncate_content(redacted)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _redact_secrets(self, content: Any) -> Any:
        """Replace high-confidence secrets while preserving surrounding content."""
        if isinstance(content, str):
            redacted = content
            for pattern in SECRET_PATTERNS:
                redacted = pattern.sub(self._replace_secret, redacted)
            return redacted if redacted != content else content

        if not isinstance(content, list):
            return content

        changed = False
        redacted_blocks: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted = self._redact_secrets(block)
                changed = changed or redacted is not block
                redacted_blocks.append(redacted)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                redacted = self._redact_secrets(block["text"])
                changed = changed or redacted is not block["text"]
                redacted_blocks.append({**block, "text": redacted})
            else:
                redacted_blocks.append(block)
        return redacted_blocks if changed else content

    @staticmethod
    def _replace_secret(match: re.Match[str]) -> str:
        secret = match.group("secret")
        if IngressGuardsMiddleware._is_placeholder(secret):
            return match.group(0)
        return match.group(0).replace(secret, "<REDACTED_SECRET>", 1)

    @staticmethod
    def _is_placeholder(secret: str) -> bool:
        normalized = secret.strip()
        return (
            normalized.lower() == "your-password"
            or normalized.upper() == "YOUR_API_KEY_HERE"
            or normalized.lower() == "xxxxxx"
            or (normalized.startswith("<") and normalized.endswith(">"))
            or (normalized.startswith("${") and normalized.endswith("}"))
            or len(set(normalized)) <= 2
        )

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
