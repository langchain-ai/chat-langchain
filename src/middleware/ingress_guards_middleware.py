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

CREDENTIAL_PATTERNS = [
    re.compile(r"lsv2_(?:sk|pt)_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{24,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)\bAuthorization\s*:\s*[^\s,;]+(?:\s+[^\s,;]+)?"),
    re.compile(r"(?i)\bBearer\s+[^\s,;]+"),
]


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

    def _redact_content(self, content: Any) -> Any:
        """Replace credential patterns in text while preserving content blocks."""
        if isinstance(content, str):
            return self._redact_text(content)

        if not isinstance(content, list):
            return content

        changed = False
        redacted: list[Any] = []
        for block in content:
            if isinstance(block, str):
                text = self._redact_text(block)
                changed = changed or text != block
                redacted.append(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = self._redact_text(block["text"])
                changed = changed or text != block["text"]
                redacted.append({**block, "text": text})
            else:
                redacted.append(block)
        return redacted if changed else content

    def _redact_text(self, text: str) -> str:
        """Replace non-placeholder credential matches in text."""
        for pattern in CREDENTIAL_PATTERNS:
            text = pattern.sub(self._replace_credential, text)
        return text

    @staticmethod
    def _replace_credential(match: re.Match[str]) -> str:
        """Return a placeholder unless the match is already non-secret text."""
        value = match.group(0).lower()
        if any(
            marker in value
            for marker in ("your_", "<", "xxxx", "example", "[redacted_credential]")
        ):
            return match.group(0)
        return "[REDACTED_CREDENTIAL]"

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


__all__ = ["CREDENTIAL_PATTERNS", "IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
