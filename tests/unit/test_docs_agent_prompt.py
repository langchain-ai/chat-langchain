"""Tests for the docs agent prompt scope guidance."""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_scope_guidance_preserves_in_scope_carve_outs():
    scope_line = next(
        line for line in docs_agent_prompt.splitlines() if line.startswith("**Scope:")
    )
    scope = scope_line.lower()

    assert "language help" not in scope
    assert "reply in or translate the answer into the user's own language" in scope
    assert "always in scope" in scope
    assert "scope has already been decided" in scope
    assert "do not re-adjudicate" in scope
    assert "guardrails classifier's allowed decision is authoritative" in scope
    assert "refuse it by repeating this scope paragraph" in scope
    assert "build, design, or explain" in scope
    assert "agent, tool, or middleware" in scope
    assert "regardless of the application domain" in scope
    assert "test generation" in scope
    assert "data science" in scope
    assert "a calculator" in scope
    assert "own scope, capabilities, or identity" in scope
    assert "bare term lookup" in scope
    assert "term was not found in the langchain documentation" in scope
    assert "closest documented match" in scope
    assert "general knowledge" in scope
