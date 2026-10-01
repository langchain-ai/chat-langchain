"""Root-run LangSmith metadata for the managed docs agent.

Pass this dict to ``define_deep_agent(metadata=...)`` so the runtime applies
``.with_config`` after ``create_deep_agent`` — nested middleware cannot
reliably reach the LangSmith root run tree.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.utils.prompt_provenance import get_prompt_provenance

_PROVENANCE_GRAPH_ID = "docs_agent"
logger = logging.getLogger(__name__)


def _running_git_sha() -> str | None:
    """Return the SHA identifying the running source revision."""
    revision = os.environ.get("LANGCHAIN_REVISION_ID") or os.environ.get(
        "LANGSMITH_HOST_REVISION_ID"
    )
    if revision:
        return revision
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(value: str) -> str:
    """Return the SHA-256 digest of prompt text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_hashes() -> dict[str, str]:
    """Hash the prompt text resolved by the runtime."""
    docs_prompt = Path(__file__).resolve().parents[2] / "instructions.md"
    guardrails_module = sys.modules.get("src.middleware.guardrails_middleware")
    guardrails_prompt = getattr(guardrails_module, "_GUARDRAILS_SYSTEM_PROMPT", None)
    if guardrails_prompt is None:
        from src.prompts.guardrails_prompts import guardrails_system_prompt

        guardrails_prompt = guardrails_system_prompt
    return {
        "docs_agent_prompt_sha256": _sha256(docs_prompt.read_text()),
        "guardrails_prompt_sha256": _sha256(guardrails_prompt),
    }


def build_docs_agent_trace_metadata(
    *,
    graph_id: str = _PROVENANCE_GRAPH_ID,
) -> dict[str, str]:
    """Return metadata that should land on the root LangSmith run."""
    metadata: dict[str, str] = {
        "source_type": "Chat-LangChain",
        **get_prompt_provenance(graph_id),
        **_prompt_hashes(),
    }
    revision = _running_git_sha()
    if revision:
        metadata["LANGSMITH_AGENT_VERSION"] = revision
        metadata["git_sha"] = revision
    logger.info(
        "Running agent revision=%s docs_agent_prompt_sha256=%s guardrails_prompt_sha256=%s",
        revision or "unknown",
        metadata["docs_agent_prompt_sha256"],
        metadata["guardrails_prompt_sha256"],
    )
    return metadata


__all__ = ["build_docs_agent_trace_metadata"]
