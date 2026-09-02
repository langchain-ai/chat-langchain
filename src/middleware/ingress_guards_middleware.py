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

import copy
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000
SECRET_PATTERNS = [
    re.compile(r"(?P<langsmith>lsv2_(?:pt|sk)_[0-9a-f]{16,}_[0-9a-f]{6,})"),
    re.compile(r"(?P<openai>sk-[A-Za-z0-9]{20,})"),
    re.compile(r"(?P<google>AIza[0-9A-Za-z_-]{30,})"),
    re.compile(r"(?P<github>gh[pousr]_[A-Za-z0-9]{30,})"),
    re.compile(r"(?P<slack>xox[baprs]-[A-Za-z0-9-]{10,})"),
    re.compile(r"(?P<bearer>Bearer [A-Za-z0-9._-]{24,})"),
]

_SECRET_PLACEHOLDERS = {
    "langsmith": "<REDACTED_API_KEY>",
    "openai": "<REDACTED_API_KEY>",
    "google": "<REDACTED_API_KEY>",
    "github": "<REDACTED_API_KEY>",
    "slack": "<REDACTED_API_KEY>",
    "bearer": "<REDACTED_BEARER_TOKEN>",
}
logger = logging.getLogger(__name__)


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Redact secrets and truncate the latest user message."""
        messages = state.get("messages", [])
        latest_human = next(
            (
                message
                for message in reversed(messages)
                if getattr(message, "type", None) == "human"
            ),
            None,
        )
        redacted_count = 0
        changed_messages = []
        for message in messages:
            content, count = self._redact_content(message.content)
            redacted_count += count
            if message is latest_human:
                content = self._truncate_content(content)
            if content is not message.content:
                sanitized = copy.copy(message)
                sanitized.content = content
                changed_messages.append(sanitized)

        if redacted_count:
            logger.warning(
                "redacted secrets at agent ingress",
                extra={"redaction_count": redacted_count},
            )
        return {"messages": changed_messages} if changed_messages else None

    def _redact_content(self, content: Any) -> tuple[Any, int]:
        """Replace high-confidence credential patterns in message text."""
        if isinstance(content, str):
            return _redact_secrets(content)

        if not isinstance(content, list):
            return content, 0

        changed = False
        redacted_count = 0
        redacted: list[Any] = []
        for block in content:
            if isinstance(block, str):
                sanitized, count = _redact_secrets(block)
                redacted.append(sanitized)
                changed = changed or count > 0
                redacted_count += count
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                sanitized, count = _redact_secrets(block["text"])
                redacted.append({**block, "text": sanitized} if count else block)
                changed = changed or count > 0
                redacted_count += count
            else:
                redacted.append(block)
        return (redacted if changed else content), redacted_count

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


def _redact_secrets(text: str) -> tuple[str, int]:
    redaction_count = 0
    for pattern in SECRET_PATTERNS:
        text, count = pattern.subn(
            lambda match: _SECRET_PLACEHOLDERS[match.lastgroup or ""], text
        )
        redaction_count += count
    return text, redaction_count


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS", "SECRET_PATTERNS"]
