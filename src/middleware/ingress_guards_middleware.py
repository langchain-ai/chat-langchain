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
        r"(?P<scheme>postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|https)://"
        r"(?P<user>[^\s:/@]+):(?P<pw>[^\s@/]+)@",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>sk-ant-|sk-|lsv2_|lcl_|AIza|ghp_|xoxb-|tvly-|pk_live_|AKIA)"
        r"(?P<secret>[^\s'\"`<>]+)"
    ),
    re.compile(
        r"(?P<name>(?:api[_-]?key|token|secret|password|passwd|pwd))"
        r"(?P<separator>\s*=\s*)(?P<secret>[^\s'\"`<>]+)",
        re.IGNORECASE,
    ),
]


def _is_placeholder(value: str) -> bool:
    """Return whether a value is an obvious documentation placeholder."""
    normalized = value.strip().lower()
    return (
        normalized == "password"
        or normalized.startswith("<your")
        or normalized.startswith("your_")
        or normalized.startswith("xxx")
        or normalized.startswith("${")
        or normalized.startswith("os.getenv(")
    )


def _redact_text(text: str) -> str:
    """Redact credential-like spans from user text."""
    def replace(match: re.Match[str]) -> str:
        if match.groupdict().get("pw") is not None:
            password = match.group("pw")
            if _is_placeholder(password):
                return match.group(0)
            return (
                f"{match.group('scheme')}://{match.group('user')}:"
                "[REDACTED_PASSWORD]@"
            )
        secret = match.group("secret")
        if _is_placeholder(secret):
            return match.group(0)
        if match.groupdict().get("prefix") is not None:
            return f"{match.group('prefix')}[REDACTED_SECRET]"
        return f"{match.group('name')}{match.group('separator')}[REDACTED_SECRET]"

    for pattern in REDACTION_PATTERNS:
        text = pattern.sub(replace, text)
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
                capped = self._truncate_content(message.content)
                if capped != message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _truncate_content(self, content: Any) -> Any:
        """Redact and trim user text while preserving non-text content blocks."""
        if isinstance(content, str):
            redacted = _redact_text(content)
            return redacted[:MAX_MESSAGE_CHARS] if len(redacted) > MAX_MESSAGE_CHARS else redacted

        if not isinstance(content, list):
            return content

        remaining = MAX_MESSAGE_CHARS
        changed = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted = _redact_text(block)
                text = redacted[:remaining]
                changed = changed or text != block
                truncated.append(text)
                remaining -= len(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                redacted = _redact_text(block["text"])
                text = redacted[:remaining]
                changed = changed or text != block["text"]
                truncated.append({**block, "text": text})
                remaining -= len(text)
            else:
                truncated.append(block)
        return truncated if changed else content


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
