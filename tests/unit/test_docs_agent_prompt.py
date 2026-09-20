from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_declined_turn_does_not_stick_to_new_allowed_docs_question():
    prompt_lower = docs_agent_prompt.lower()

    assert "classify every later turn independently" in prompt_lower
    assert "repeats or clearly rewords that same declined request" in prompt_lower
    assert "new allowed or ordinary potentially in-scope question" in prompt_lower
    assert "normal bounded search workflow" in prompt_lower
    assert (
        "not with the fixed scope-refusal sentence merely because an earlier turn was refused"
        in prompt_lower
    )
    assert "langchane docs" in prompt_lower
    assert "plz ful docs" in prompt_lower
