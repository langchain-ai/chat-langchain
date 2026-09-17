"""Prompt provenance lookup for LangSmith trace metadata."""

import hashlib
import logging
import os
from functools import lru_cache

from src.prompts.guardrails_prompts import guardrails_system_prompt

logger = logging.getLogger(__name__)

#: Workspace that owns the Hub prompts used for ``prompt_source`` /
#: ``prompt_commit`` provenance. On MDA deploys this is often a different
#: workspace than ``LANGSMITH_WORKSPACE_ID`` (the deployment's own org).
_PROMPT_WORKSPACE_ENV = "LANGSMITH_PROMPT_WORKSPACE_ID"

#: Optional API key for Hub provenance pulls. ``LANGSMITH_API_KEY`` is reserved
#: by MDA deploy and replaced by the host-injected key, which may not be able to
#: read prompts in ``LANGSMITH_PROMPT_WORKSPACE_ID``. Set this to a key that can.
_PROMPT_API_KEY_ENV = "LANGSMITH_PROMPT_API_KEY"

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


def _prompt_api_key() -> str | None:
    """Return an API key that can read the Hub prompt workspace, if configured."""
    value = os.getenv(_PROMPT_API_KEY_ENV, "").strip()
    return value or None


def _hub_client(workspace_id: str | None, api_key: str | None):
    """Build a LangSmith client scoped for Hub provenance reads."""
    from langsmith import Client

    kwargs: dict[str, str] = {}
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    if api_key:
        kwargs["api_key"] = api_key
    return Client(**kwargs)


@lru_cache(maxsize=16)
def _resolve_hub_provenance(
    hub_name: str,
    workspace_id: str | None,
    api_key_set: bool,
) -> tuple[str, str | None]:
    """Pull the hub prompt to get its current commit hash. Cached per process.

    ``api_key_set`` is part of the cache key so enabling ``LANGSMITH_PROMPT_API_KEY``
    invalidates prior failed lookups. The key value itself is read at call time.
    """
    try:
        client = _hub_client(workspace_id, _prompt_api_key() if api_key_set else None)
        template = client.pull_prompt(hub_name)
        commit = (template.metadata or {}).get("lc_hub_commit_hash")
        if not commit:
            logger.warning(
                "Hub prompt %s pulled but lc_hub_commit_hash missing (workspace=%s)",
                hub_name,
                workspace_id or "default",
            )
        return f"hub:{hub_name}", commit
    except Exception as exc:
        logger.warning(
            "Failed to resolve hub provenance for %s (workspace=%s, prompt_api_key=%s): %s",
            hub_name,
            workspace_id or "default",
            "set" if api_key_set else "unset",
            exc,
        )
        return f"hub:{hub_name}", None


@lru_cache(maxsize=16)
def _resolve_guardrails_provenance(
    hub_name: str,
    workspace_id: str | None,
    api_key_set: bool,
) -> tuple[str, str | None, str | None, bool | None]:
    """Pull guardrails prompt provenance and compare its content to the repo."""
    source = f"hub:{hub_name}"
    try:
        client = _hub_client(workspace_id, _prompt_api_key() if api_key_set else None)
        template = client.pull_prompt(hub_name)
        commit = (template.metadata or {}).get("lc_hub_commit_hash")
        try:
            prompt_text = template.format_messages(messages=[])[0].content
        except (AttributeError, IndexError, TypeError):
            return source, commit, None, None
        prompt_sha256 = hashlib.sha256(prompt_text.encode()).hexdigest()
        repo_sha256 = hashlib.sha256(guardrails_system_prompt.encode()).hexdigest()
        return source, commit, prompt_sha256, prompt_sha256 == repo_sha256
    except Exception as exc:
        logger.warning(
            "Failed to resolve guardrails prompt content for %s (workspace=%s): %s",
            hub_name,
            workspace_id or "default",
            exc,
        )
        return source, None, None, None


def get_prompt_provenance(graph_id: str) -> dict[str, str | bool]:
    """Return prompt provenance for a graph_id."""
    if _USE_LOCAL_PROMPTS and graph_id == "docs_agent":
        return {
            "prompt_source": "local:instructions.md",
            "guardrails_prompt_source": "local:src/prompts/guardrails_prompts.py",
            "guardrails_prompt_matches_repo": True,
            "guardrails_prompt_sha256": hashlib.sha256(
                guardrails_system_prompt.encode()
            ).hexdigest(),
        }

    if graph_id in _HUB_PROMPTS:
        workspace_id = _prompt_workspace_id()
        api_key_set = _prompt_api_key() is not None
        source, commit = _resolve_hub_provenance(
            _HUB_PROMPTS[graph_id], workspace_id, api_key_set
        )
        (
            guardrails_source,
            guardrails_commit,
            guardrails_sha256,
            guardrails_matches_repo,
        ) = _resolve_guardrails_provenance(
            _GUARDRAILS_HUB_PROMPTS[graph_id], workspace_id, api_key_set
        )
        provenance = {
            "prompt_source": source,
            "guardrails_prompt_source": guardrails_source,
        }
        if commit:
            provenance["prompt_commit"] = commit
        if guardrails_commit:
            provenance["guardrails_prompt_commit"] = guardrails_commit
        if guardrails_sha256:
            provenance["guardrails_prompt_sha256"] = guardrails_sha256
        if guardrails_matches_repo is not None:
            provenance["guardrails_prompt_matches_repo"] = guardrails_matches_repo

        return provenance

    return {}
