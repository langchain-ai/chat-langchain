"""Ingress guards for Chat LangChain on Managed Deep Agents.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware caps oversized user input and redacts credentials in user text.

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

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


_SECRET_RE = re.compile(
    r"(?P<label>\b[\w-]*(?:key|token|secret|password)[\"']?\s*[:=]\s*)"
    r"(?!\[REDACTED_SECRET\])(?:\"[^\"\n]+\"|'[^'\n]+'|[^\s&\#\"',;)}\]]+)"
    r"|\b(?:sk-(?:ant-)?|tvly-|AIza|gh[po]_|xoxb-|pk_live_|lsv2_|lcl_)[A-Za-z0-9_-]+"
    r"|\bBearer[ \t]+[A-Za-z0-9._~+/=-]+"
    r"|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    re.IGNORECASE,
)


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap user input and redact credentials at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Cap and redact the latest user message before downstream processing."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                redacted = self._redact_content(capped)
                if redacted != capped:
                    self._track_redaction_metadata()
                if capped is not message.content or redacted != capped:
                    # Same id => the messages reducer overwrites in place.
                    message.content = redacted
                    return {"messages": [message]}
                break
        return None

    def _redact_content(self, content: Any) -> Any:
        """Replace credentials in text while preserving non-text blocks."""
        if isinstance(content, str):
            return _SECRET_RE.sub(
                lambda match: (match.group("label") or "") + "[REDACTED_SECRET]",
                content,
            )
        if not isinstance(content, list):
            return content
        return [
            self._redact_content(block)
            if isinstance(block, str)
            else {**block, "text": self._redact_content(block["text"])}
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            else block
            for block in content
        ]

    def _track_redaction_metadata(self) -> None:
        """Mark runs whose latest user message contained credentials."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree:
                run_tree.metadata["secret_redacted"] = True
        except Exception:
            pass

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
