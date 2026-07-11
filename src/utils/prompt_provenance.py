# Prompt provenance lookup for LangSmith trace metadata.

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

#: Workspace that owns the Hub prompts used for ``prompt_source`` /
#: ``prompt_commit`` provenance. On MDA deploys this is often a different
#: workspace than ``LANGSMITH_WORKSPACE_ID`` (the deployment's own org).
_PROMPT_WORKSPACE_ENV = "LANGSMITH_PROMPT_WORKSPACE_ID"

_USE_LOCAL_PROMPTS = os.getenv("USE_LOCAL_PROMPTS", "").lower() in {
    "1",
    "true",
    "yes",
}
_USE_STAGING = (
    os.getenv("LANGSMITH_HOST_PROJECT_NAME") == "immanuel-chat-langchain-test"
    or os.getenv("LANGSMITH_ENV") == "dev"
)
_HUB_PROMPTS: dict[str, str] = {
    "docs_agent": (
        "public-chat-langchain-test:staging"
        if _USE_STAGING
        else "public-chat-langchain-test:production"
    ),
}
_GUARDRAILS_HUB_PROMPTS: dict[str, str] = {
    "docs_agent": (
        "public-chat-langchain-guardrails-test:staging"
        if _USE_STAGING
        else "public-chat-langchain-guardrails-test:production"
    ),
}


def _prompt_workspace_id() -> str | None:
    """Return the Hub-owning workspace id, if configured."""
    value = os.getenv(_PROMPT_WORKSPACE_ENV, "").strip()
    return value or None


@lru_cache(maxsize=16)
def _resolve_hub_provenance(
    hub_name: str, workspace_id: str | None
) -> tuple[str, str | None]:
    """Pull the hub prompt to get its current commit hash. Cached per process."""
    try:
        from langsmith import Client

        client = Client(workspace_id=workspace_id) if workspace_id else Client()
        template = client.pull_prompt(hub_name)
        commit = (template.metadata or {}).get("lc_hub_commit_hash")
        return f"hub:{hub_name}", commit
    except Exception as exc:
        logger.warning(
            "Failed to resolve hub provenance for %s (workspace=%s): %s",
            hub_name,
            workspace_id or "default",
            exc,
        )
        return f"hub:{hub_name}", None


def get_prompt_provenance(graph_id: str) -> dict[str, str]:
    """Return prompt provenance for a graph_id."""
    if _USE_LOCAL_PROMPTS and graph_id == "docs_agent":
        return {
            "prompt_source": "local:src/prompts/docs_agent_prompt.py",
            "guardrails_prompt_source": "local:src/prompts/guardrails_prompts.py",
        }

    if graph_id in _HUB_PROMPTS:
        workspace_id = _prompt_workspace_id()
        source, commit = _resolve_hub_provenance(_HUB_PROMPTS[graph_id], workspace_id)
        guardrails_source, guardrails_commit = _resolve_hub_provenance(
            _GUARDRAILS_HUB_PROMPTS[graph_id], workspace_id
        )
        provenance = {
            "prompt_source": source,
            "guardrails_prompt_source": guardrails_source,
        }
        if commit:
            provenance["prompt_commit"] = commit
        if guardrails_commit:
            provenance["guardrails_prompt_commit"] = guardrails_commit

        return provenance

    return {}
