from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_prompt_requires_not_found_answer_for_unsupported_entity():
    """Require an explicit not-found lead when retrieval omits the named entity."""
    question = "Does a2a have auth?"
    retrieval_results = "Authentication configuration for LangChain agents."
    prompt_lower = docs_agent_prompt.lower()

    assert "first search query" in prompt_lower
    assert "named" in prompt_lower and "verbatim" in prompt_lower
    assert "appears literally in at least one tool result" in prompt_lower
    assert "documentation contains nothing about that name" in prompt_lower
    assert "different thing" in prompt_lower
    assert "invented origin, package, or product attribution" in prompt_lower
    assert question.split()[1].lower() in question.lower()
    assert "a2a" not in retrieval_results.lower()
