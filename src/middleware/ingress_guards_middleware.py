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

Trace metadata is stamped on the **root** run (walking ``parent_run``), matching
legacy ``docs_graph.py``'s ``with_config(metadata=...)`` so Hub provenance and
``LANGSMITH_AGENT_VERSION`` remain filterable at the trace root. Git-linked
host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are not synthesized here — MDA
archive deploys use ``LANGSMITH_HOST_REVISION_ID`` / ``LANGSMITH_AGENT_VERSION``
instead.
"""

from __future__ import annotations

import logging
import os
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


def _root_run_tree(run_tree: Any) -> Any:
    """Walk to the outermost parent so metadata lands on the trace root."""
    root = run_tree
    while getattr(root, "parent_run", None) is not None:
        root = root.parent_run
    return root


def _agent_version_metadata() -> dict[str, str]:
    """Mirror legacy docs_graph stamping of ``LANGSMITH_AGENT_VERSION``."""
    revision = os.environ.get("LANGCHAIN_REVISION_ID") or os.environ.get(
        "LANGSMITH_HOST_REVISION_ID"
    )
    if not revision:
        return {}
    return {"LANGSMITH_AGENT_VERSION": revision}


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input and stamp Chat LangChain trace metadata."""

    def __init__(self, provenance_graph_id: str = _PROVENANCE_GRAPH_ID) -> None:
        super().__init__()
        # Resolve Hub provenance once per process (same idea as docs_graph import
        # time), so request paths do not re-pull and commits are stable.
        from src.utils.prompt_provenance import get_prompt_provenance

        self._provenance = get_prompt_provenance(provenance_graph_id)

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
        """Stamp source_type, prompt provenance, and agent version on the root run."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree is None:
                return
            root = _root_run_tree(run_tree)
            metadata = getattr(root, "metadata", None)
            if not isinstance(metadata, dict):
                return

            metadata.setdefault("source_type", "Chat-LangChain")
            for key, value in self._provenance.items():
                metadata.setdefault(key, value)
            for key, value in _agent_version_metadata().items():
                metadata.setdefault(key, value)
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
