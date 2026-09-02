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
REDACTED_SECRET = "YOUR_API_KEY_HERE"

CREDENTIAL_PATTERNS = [
    re.compile(
        r"(?<![A-Za-z0-9_])(?:lsv2_pt_|lsv2_sk_|lcl_|sk-proj-|sk-|tvly-|AIza|ghp_|gho_|xoxb-|pk_live_)(?!your_api_key_here\b)[A-Za-z0-9_.-]+"
    ),
    re.compile(r"(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+"),
    re.compile(
        r"(?i)(?P<prefix>(?:[A-Za-z0-9_]*)?(?:api_key|token|secret|password)\s*=\s*)(?P<quote>[\"']?)(?!lsv2_your_api_key_here\b)(?P<value>[^\"'\s,;&}]+)(?P=quote)"
    ),
    re.compile(
        r"(?i)(?P<prefix>X-Api-Key\s*:\s*)(?P<quote>[\"']?)(?!lsv2_your_api_key_here\b)(?P<value>[^\"'\s,;]+)(?P=quote)"
    ),
    re.compile(
        r"(?i)(?P<prefix>Authorization\s*:\s*Bearer\s+)(?P<quote>[\"']?)(?!lsv2_your_api_key_here\b)(?P<value>[^\"'\s,;]+)(?P=quote)"
    ),
]


def _redact_secrets(text: str) -> str:
    """Replace credential values with a stable placeholder."""
    for pattern in CREDENTIAL_PATTERNS:
        text = pattern.sub(
            lambda match: (
                f"{match.group('prefix')}{match.group('quote')}{REDACTED_SECRET}{match.group('quote')}"
                if "prefix" in match.groupdict()
                else REDACTED_SECRET
            ),
            text,
        )
    return text


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message when it exceeds the size cap."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _truncate_content(self, content: Any) -> Any:
        """Trim user text to the cap while preserving non-text content blocks."""
        if isinstance(content, str):
            redacted = _redact_secrets(content)
            return (
                redacted[:MAX_MESSAGE_CHARS]
                if len(redacted) > MAX_MESSAGE_CHARS
                else redacted
            )

        if not isinstance(content, list):
            return content

        remaining = MAX_MESSAGE_CHARS
        changed = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                redacted = _redact_secrets(block)
                text = redacted[:remaining]
                changed = changed or text != block
                truncated.append(text)
                remaining -= len(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                redacted = _redact_secrets(block["text"])
                text = redacted[:remaining]
                changed = changed or text != block["text"]
                truncated.append({**block, "text": text})
                remaining -= len(text)
            else:
                truncated.append(block)
        return truncated if changed else content


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
