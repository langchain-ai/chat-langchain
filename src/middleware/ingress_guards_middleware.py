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
_URI_CREDENTIAL = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.\-]*://)(?P<user>[^\s:@/]{1,256}):"
    r"(?P<secret>[^\s@/]{1,512})@",
    re.IGNORECASE,
)
_KEY_VALUE_SECRET = re.compile(
    r"(?P<name>\b(?:api_key|token|secret|password)\s*=\s*)(?P<secret>[^\s&;,]+)",
    re.IGNORECASE,
)
_PREFIX_SECRET = re.compile(
    r"(?<![\w-])(?P<secret>(?:sk-|tvly-|AIza|ghp_|xoxb-|pk_live_|lsv2_|lcl_)[^\s'\"`<>\]}),;]+)",
)
_BEARER_SECRET = re.compile(
    r"(?P<prefix>\bBearer\s+)(?P<secret>[^\s'\"`<>\]}),;]+)", re.IGNORECASE
)
_JWT_SECRET = re.compile(
    r"(?<![\w-])(?P<secret>[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)(?![\w-])"
)


def _is_placeholder(secret: str) -> bool:
    """Return whether a value is an obvious non-secret placeholder."""
    return secret in {"<password>", "${PGPASSWORD}"} or (
        len(secret) > 1 and set(secret) <= {"x", "X", "*"}
    )


def _replace_secret(match: re.Match[str]) -> str:
    secret = match.group("secret")
    if _is_placeholder(secret):
        return match.group(0)
    return match.group(0).replace(secret, "[REDACTED_SECRET]", 1)


def _redact_text(text: str) -> str:
    """Redact credentials from text while preserving surrounding content."""
    for pattern in (
        _URI_CREDENTIAL,
        _KEY_VALUE_SECRET,
        _PREFIX_SECRET,
        _BEARER_SECRET,
        _JWT_SECRET,
    ):
        text = pattern.sub(_replace_secret, text)
    return text


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact and truncate the latest user message."""
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
        """Redact secrets from text content while preserving non-text blocks."""
        if isinstance(content, str):
            redacted = _redact_text(content)
            return redacted if redacted != content else content

        if not isinstance(content, list):
            return content

        changed = False
        redacted_content: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted = _redact_text(block)
                changed = changed or redacted != block
                redacted_content.append(redacted)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = block["text"]
                redacted = _redact_text(text)
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
