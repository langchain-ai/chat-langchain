"""Tests for the docs agent's evidence-grounding prompt rules."""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_specific_entities_are_searched_verbatim_alongside_concepts():
    """Keep named entities in dedicated documentation queries."""
    assert '`query="a2a"` + `query="authentication"`' in docs_agent_prompt
    assert (
        '`query="create_async_playwright_browser"` + `query="browser"`'
        in docs_agent_prompt
    )
    assert "Cache optimization must not remove the entity being asked about." in docs_agent_prompt


def test_missing_entities_require_an_evidence_based_response():
    """Require an explicit documentation gap for unsupported entities."""
    assert "documentation does not cover that name" in docs_agent_prompt
    assert "must not define, import, call, or provide runnable code" in docs_agent_prompt
