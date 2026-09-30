"""Tests for Deep Agents documentation routing guidance."""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_deep_agents_library_queries_keep_surface_and_language_context():
    """Deep Agents library questions must not be reduced to an ambiguous noun."""
    prompt = docs_agent_prompt.lower()

    assert "deepagents library" in prompt
    assert "create_deep_agent" in prompt
    assert "retain `python` or `javascript`" in prompt
    assert "if relying on library pages, explicitly name the `deepagents` library" in prompt

    library_query = "deepagents library"
    answer_lead = "**The Python `deepagents` library uses `create_deep_agent`.**"
    assert "library" in library_query
    assert "dcode" not in answer_lead.lower()
