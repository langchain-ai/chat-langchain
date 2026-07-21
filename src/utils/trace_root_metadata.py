"""Root-run LangSmith metadata for the managed docs agent.

Pass this dict to ``define_deep_agent(metadata=...)`` so the runtime applies
``.with_config`` after ``create_deep_agent`` — nested middleware cannot
reliably reach the LangSmith root run tree.
"""

from __future__ import annotations

import os

from src.utils.prompt_provenance import get_prompt_provenance

_PROVENANCE_GRAPH_ID = "docs_agent"


def build_docs_agent_trace_metadata(
    *,
    graph_id: str = _PROVENANCE_GRAPH_ID,
) -> dict[str, str]:
    """Return metadata that should land on the root LangSmith run."""
    metadata: dict[str, str] = {
        "source_type": "Chat-LangChain",
        **get_prompt_provenance(graph_id),
    }
    revision = os.environ.get("LANGCHAIN_REVISION_ID") or os.environ.get(
        "LANGSMITH_HOST_REVISION_ID"
    )
    if revision:
        metadata["LANGSMITH_AGENT_VERSION"] = revision
    return metadata


__all__ = ["build_docs_agent_trace_metadata"]
