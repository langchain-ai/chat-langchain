"""Input-side credential redaction middleware.

Scans incoming human messages for high-entropy credential patterns and
rewrites them to typed placeholders BEFORE the message reaches the model
or the trace logger. Output-side scrubbing is not sufficient: once the
raw secret enters the model context it is already persisted in the trace.
"""

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# More specific patterns must come before broader ones (e.g. github_pat_ before ghp_).
_CREDENTIAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "github_pat_***REDACTED***"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "ghp_***REDACTED***"),
    (re.compile(r"AKIA[A-Z0-9]{16}"), "AKIA***REDACTED***"),
    (re.compile(r"xoxp-[A-Za-z0-9-]{10,}"), "xoxp-***REDACTED***"),
    (re.compile(r"sk-[A-Za-z0-9]{32,}"), "sk-***REDACTED***"),
    (
        re.compile(r"ey[A-Za-z0-9_-]{10,}\.ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "jwt.***REDACTED***",
    ),
]


def _redact(text: str) -> tuple[str, list[str]]:
    """Apply each credential pattern; return (redacted_text, matched_patterns)."""
    matched: list[str] = []
    for pattern, placeholder in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            matched.append(pattern.pattern)
            text = pattern.sub(placeholder, text)
    return text, matched


class CredentialRedactionMiddleware(AgentMiddleware[AgentState]):
    """Redact credentials in incoming human messages before the model sees them."""

    def before_agent(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Rewrite HumanMessages whose content contains credential patterns."""
        messages = state.get("messages") or []
        updated: list[Any] = []
        any_redacted = False
        for msg in messages:
            if not isinstance(msg, HumanMessage):
                updated.append(msg)
                continue
            content = msg.content
            if not isinstance(content, str):
                updated.append(msg)
                continue
            redacted, matched = _redact(content)
            if matched:
                any_redacted = True
                metadata = dict(getattr(msg, "additional_kwargs", {}) or {})
                metadata["credentials_redacted"] = True
                metadata["patterns_matched"] = matched
                updated.append(
                    HumanMessage(
                        content=redacted, additional_kwargs=metadata, id=msg.id
                    )
                )
                logger.warning(
                    "Redacted %d credential pattern(s) from user message: %s",
                    len(matched),
                    matched,
                )
            else:
                updated.append(msg)
        if not any_redacted:
            return None
        return {"messages": updated}
