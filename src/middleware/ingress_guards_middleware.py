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

_LANGSMITH_KEY_PATTERN = re.compile(r"lsv2_(?:sk|pt)_[A-Za-z0-9_]{16,}")
_OPENAI_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
_GITHUB_TOKEN_PATTERN = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")
_AWS_ACCESS_KEY_PATTERN = re.compile(r"AKIA[0-9A-Z]{16}")
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._-]{20,})")
_PASSWORD_ASSIGNMENT_PATTERN = re.compile(
    r'''(?i)\b(password|passwd|pwd)(\s*[:=]\s*|["']\s*:\s*)["']?([^\s"'<>,;]{6,})'''
)
_URI_PASSWORD_PATTERN = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:([^@\s/]+)@"
)
_PATH_IDENTIFIER_PATTERN = re.compile(
    r"(^|[/\\])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?=[/\\]|$)"
)


def _is_placeholder(value: str) -> bool:
    """Return whether a value is already a recognizable placeholder."""
    lowered = value.lower()
    return any(marker in lowered for marker in ("your_", "<", "xxx", "...", "example"))


def _redact_secrets(text: str) -> str:
    """Replace high-confidence secrets and path identifiers with placeholders."""
    for pattern, placeholder in (
        (_LANGSMITH_KEY_PATTERN, "<REDACTED_API_KEY>"),
        (_OPENAI_KEY_PATTERN, "<REDACTED_API_KEY>"),
        (_GITHUB_TOKEN_PATTERN, "<REDACTED_TOKEN>"),
        (_AWS_ACCESS_KEY_PATTERN, "<REDACTED_API_KEY>"),
    ):
        text = pattern.sub(
            lambda match: match.group(0)
            if _is_placeholder(match.group(0))
            else placeholder,
            text,
        )

    text = _BEARER_TOKEN_PATTERN.sub(
        lambda match: match.group(1)
        if _is_placeholder(match.group(2))
        else f"{match.group(1)}<REDACTED_TOKEN>",
        text,
    )
    text = _PASSWORD_ASSIGNMENT_PATTERN.sub(
        lambda match: match.group(0)
        if _is_placeholder(match.group(3))
        else f"{match.group(1)}{match.group(2)}<REDACTED_PASSWORD>",
        text,
    )
    text = _URI_PASSWORD_PATTERN.sub(
        lambda match: match.group(0)
        if _is_placeholder(match.group(1))
        else match.group(0).replace(match.group(1), "<REDACTED_PASSWORD>"),
        text,
    )
    return _PATH_IDENTIFIER_PATTERN.sub(
        lambda match: match.group(0)
        if _is_placeholder(match.group(2))
        else f"{match.group(1)}<REDACTED_PERSONAL_IDENTIFIER>",
        text,
    )


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message when it exceeds the size cap."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._sanitize_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _sanitize_content(self, content: Any) -> Any:
        """Redact secrets and trim user text while preserving content blocks."""
        if isinstance(content, str):
            sanitized = _redact_secrets(content)
            return sanitized[:MAX_MESSAGE_CHARS] if len(sanitized) > MAX_MESSAGE_CHARS else sanitized

        if not isinstance(content, list):
            return content

        remaining = MAX_MESSAGE_CHARS
        changed = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                sanitized = _redact_secrets(block)
                text = sanitized[:remaining]
                changed = changed or text != block
                truncated.append(text)
                remaining -= len(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                sanitized = _redact_secrets(block["text"])
                text = sanitized[:remaining]
                changed = changed or text != block["text"]
                truncated.append({**block, "text": text})
                remaining -= len(text)
            else:
                truncated.append(block)
        return truncated if changed else content


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
