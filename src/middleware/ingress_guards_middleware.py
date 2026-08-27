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
REDACTION_PATTERNS = [
    re.compile(
        r"(?P<prefix>(?:postgres|postgresql|mysql|redis|mongodb|amqp|http|https)://[^/@\s:]+:)"
        r"(?P<password>[^@\s]*)"
        r"(?P<suffix>@[^\s/?#]+(?:[/?#][^\s]*)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<key>\b(?:PASSWORD|DATABASE_URL|POSTGRES_URI_CUSTOM))="
        r"(?P<value>[^\s\"'`]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\w-])(?:sk-|tvly-|AIza|ghp_|xoxb-|pk_live_|lsv2_|lcl_)[A-Za-z0-9._~+/=-]+",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<prefix>\bBearer\s+)[^\s,;]+", re.IGNORECASE),
    re.compile(r"(?<![\w-])eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
]


def _redact_secrets(text: str) -> str:
    """Replace credentials in user-provided text with stable placeholders."""
    redacted = text
    for pattern in REDACTION_PATTERNS:
        if pattern is REDACTION_PATTERNS[0]:
            redacted = pattern.sub(r"\g<prefix><REDACTED_PASSWORD>\g<suffix>", redacted)
        elif pattern is REDACTION_PATTERNS[1]:

            def replace_assignment(match: re.Match[str]) -> str:
                value = match.group("value")
                if (
                    match.group("key").upper() != "PASSWORD"
                    and value.lower().startswith(
                        (
                            "postgres://",
                            "postgresql://",
                            "mysql://",
                            "redis://",
                            "mongodb://",
                            "amqp://",
                            "http://",
                            "https://",
                        )
                    )
                    and "<REDACTED_PASSWORD>" in value
                ):
                    return match.group(0)
                return f"{match.group('key')}=<REDACTED_PASSWORD>"

            redacted = pattern.sub(replace_assignment, redacted)
        elif pattern is REDACTION_PATTERNS[3]:
            redacted = pattern.sub(r"\g<prefix><REDACTED_CREDENTIAL>", redacted)
        else:
            redacted = pattern.sub("<REDACTED_CREDENTIAL>", redacted)
    return redacted


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message when it exceeds the size cap."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                redacted = self._redact_content(message.content)
                capped = self._truncate_content(redacted)
                if capped != message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _redact_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return _redact_secrets(content)

        if not isinstance(content, list):
            return content

        redacted: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted.append(_redact_secrets(block))
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                redacted.append({**block, "text": _redact_secrets(block["text"])})
            else:
                redacted.append(block)
        return redacted if redacted != content else content

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


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS", "_redact_secrets"]
