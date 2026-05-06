# Prompt provenance lookup for LangSmith trace metadata.

_LOCAL_PROMPT_SOURCES: dict[str, str] = {
    "docs_agent": "local:src/prompts/docs_agent_prompt.py",
}


def get_prompt_provenance(graph_id: str) -> dict[str, str]:
    """Return local prompt provenance for a graph_id."""
    if graph_id in _LOCAL_PROMPT_SOURCES:
        return {"prompt_source": _LOCAL_PROMPT_SOURCES[graph_id]}

    return {}
