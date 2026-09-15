"""Root-run LangSmith metadata for the managed docs agent.

Pass this dict to ``define_deep_agent(metadata=...)`` so the runtime applies
``.with_config`` after ``create_deep_agent`` — nested middleware cannot
reliably reach the LangSmith root run tree.
"""

from __future__ import annotations

import contextvars
import os
from collections.abc import Sequence
from typing import Any

import langsmith as ls

from src.utils.prompt_provenance import get_prompt_provenance

_PROVENANCE_GRAPH_ID = "docs_agent"
_ALLOWED_ENVIRONMENTS = frozenset({"production", "staging", "development", "preview"})
_MAX_GUARD_RETRY_COUNT = 5
_TURN_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "docs_agent_trace_turn_key", default=None
)
_GUARD_RETRY_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar(
    "docs_agent_guard_retry_count", default=0
)


def _trace_environment() -> str:
    """Return a bounded deployment environment value."""
    for variable in ("APP_ENVIRONMENT", "LANGSMITH_LANGGRAPH_API_VARIANT"):
        value = os.environ.get(variable, "").strip().lower()
        if value in _ALLOWED_ENVIRONMENTS:
            return value
    return "development"


def build_docs_agent_trace_metadata(
    *,
    graph_id: str = _PROVENANCE_GRAPH_ID,
) -> dict[str, str]:
    """Return root metadata with environment set to production, staging, development, or preview."""
    metadata: dict[str, str] = {
        "source_type": "Chat-LangChain",
        "environment": _trace_environment(),
        **get_prompt_provenance(graph_id),
    }
    revision = os.environ.get("LANGCHAIN_REVISION_ID") or os.environ.get(
        "LANGSMITH_HOST_REVISION_ID"
    )
    if revision:
        metadata["LANGSMITH_AGENT_VERSION"] = revision
    return metadata


def _turn_key(messages: Sequence[Any]) -> str:
    """Return a stable key for the latest user turn."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "id", None) or f"{index}:{message.content!r}")
    return "no-human-turn"


def update_root_run_metadata(runtime: Any, metadata: dict[str, Any]) -> None:
    """Update metadata on the LangSmith root run when one is active."""
    try:
        run_id = getattr(getattr(runtime, "execution_info", None), "run_id", None)
        if run_id:
            ls.Client().update_run(run_id, extra={"metadata": metadata})
            return

        run_tree = ls.get_current_run_tree()
        if run_tree is None:
            return
        root_run = run_tree
        while root_run.parent_run is not None:
            root_run = root_run.parent_run
        root_run.metadata.update(metadata)
    except Exception:
        return


def ensure_guard_retry_count(runtime: Any, messages: Sequence[Any]) -> int:
    """Initialize the bounded retry count for a new turn."""
    key = _turn_key(messages)
    if _TURN_KEY.get() != key:
        _TURN_KEY.set(key)
        _GUARD_RETRY_COUNT.set(0)
        update_root_run_metadata(runtime, {"guard_retry_count": 0})
    return _GUARD_RETRY_COUNT.get()


def increment_guard_retry_count(runtime: Any, messages: Sequence[Any]) -> int:
    """Increment and persist the bounded retry count for a turn."""
    ensure_guard_retry_count(runtime, messages)
    count = min(_GUARD_RETRY_COUNT.get() + 1, _MAX_GUARD_RETRY_COUNT)
    _GUARD_RETRY_COUNT.set(count)
    update_root_run_metadata(runtime, {"guard_retry_count": count})
    return count


__all__ = [
    "build_docs_agent_trace_metadata",
    "ensure_guard_retry_count",
    "increment_guard_retry_count",
    "update_root_run_metadata",
]
