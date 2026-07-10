"""Ingress guards: input caps + run-metadata enrichment.

These were previously enforced in ``src/api/auth.py`` (the ``@auth.on.threads``
``enrich_run_metadata`` callback and ``validate_inputs``). Under Managed Deep
Agents, identity/thread scoping is declared in ``identity.py``; the remaining
app-specific bits — capping oversized user input and stamping trace metadata —
live here as ordinary agent middleware.

What MDA now owns instead (and is intentionally NOT re-implemented here): token
validation, thread ownership, guest issuance, and config hygiene. Per-IP rate
limiting, the recursion-limit cap, and the ``assistant_id`` gate were dropped in
the migration (single managed graph, managed ingress).
"""

from __future__ import annotations

import logging
from typing import Any

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000

#: Logical agent id used for prompt-provenance lookup (the managed graph is named
#: "agent", but provenance is keyed on the docs agent).
_PROVENANCE_GRAPH_ID = "docs_agent"


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input and stamp Chat LangChain trace metadata."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message and enrich run metadata."""
        self._enrich_metadata()

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

    def _enrich_metadata(self) -> None:
        """Stamp source_type + prompt provenance onto the current run."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree is None:
                return
            run_tree.metadata.setdefault("source_type", "Chat-LangChain")
            from src.utils.prompt_provenance import get_prompt_provenance

            for key, value in get_prompt_provenance(_PROVENANCE_GRAPH_ID).items():
                run_tree.metadata.setdefault(key, value)
        except Exception:  # noqa: BLE001 - metadata is best-effort, never fatal
            logger.debug("Failed to stamp run metadata", exc_info=True)

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


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
