"""Prompt provenance lookup for LangSmith trace metadata."""


def _guardrails_runtime_provenance() -> tuple[str, str | None]:
    """Return the guardrails artifact and Hub commit loaded by middleware."""
    try:
        from src.middleware import guardrails_middleware

        return (
            guardrails_middleware.guardrails_prompt_source,
            guardrails_middleware.guardrails_prompt_commit,
        )
    except Exception:
        return "local:src/prompts/guardrails_prompts.py", None


def get_prompt_provenance(graph_id: str) -> dict[str, str]:
    """Return prompt provenance for a graph_id."""
    if graph_id == "docs_agent":
        provenance = {"prompt_source": "local:docs_agent_prompt"}
        source, commit = _guardrails_runtime_provenance()
        provenance["guardrails_prompt_source"] = source
        if commit:
            provenance["guardrails_prompt_commit"] = commit
        return provenance

    return {}
