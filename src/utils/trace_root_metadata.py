"""Root-run LangSmith metadata for the managed docs agent.

Pass this dict to ``define_deep_agent(metadata=...)`` so the runtime applies
``.with_config`` after ``create_deep_agent`` — nested middleware cannot
reliably reach the LangSmith root run tree.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from src.utils.prompt_provenance import get_prompt_provenance

_PROVENANCE_GRAPH_ID = "docs_agent"


def build_docs_agent_trace_metadata(
    *,
    graph_id: str = _PROVENANCE_GRAPH_ID,
    user_id: str = "anonymous",
) -> dict[str, str]:
    """Return metadata that should land on the root LangSmith run."""
    metadata: dict[str, str] = {
        "source_type": "Chat-LangChain",
        "environment": _environment_name(),
        "user_id": user_id,
        **get_prompt_provenance(graph_id),
    }
    revision = os.environ.get("LANGCHAIN_REVISION_ID") or os.environ.get(
        "LANGSMITH_HOST_REVISION_ID"
    )
    if revision:
        metadata["LANGSMITH_AGENT_VERSION"] = revision
    return metadata


def user_id_from_config(config: Mapping[str, object] | None) -> str:
    """Return the trusted per-request actor ID for root-run metadata."""
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    auth_user = (
        configurable.get("langgraph_auth_user")
        if isinstance(configurable, Mapping)
        else None
    )
    if isinstance(auth_user, Mapping):
        actor_id = auth_user.get("mda_actor_id")
        if actor_id:
            return str(actor_id)
    return "anonymous"


def _environment_name() -> str:
    explicit = os.environ.get("LANGSMITH_ENVIRONMENT") or os.environ.get("ENVIRONMENT")
    if explicit:
        return explicit

    variant = os.environ.get("LANGSMITH_LANGGRAPH_API_VARIANT", "").lower()
    git_ref = os.environ.get("LANGSMITH_LANGGRAPH_GIT_REF", "").lower()
    if any(marker in variant for marker in ("preview", "branch")):
        return "preview"
    if git_ref and git_ref not in {"main", "master", "production"}:
        return "preview"
    return "production"


__all__ = ["build_docs_agent_trace_metadata", "user_id_from_config"]
