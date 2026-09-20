from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_prompt_allows_ecosystem_teaching_and_documentation_requests():
    prompt = docs_agent_prompt.lower()

    assert "requests to learn or be taught" in prompt
    assert "locate or receive documentation" in prompt
    assert "research them with the available tools and answer them" in prompt


def test_prompt_limits_confidentiality_and_refusal_stickiness():
    prompt = docs_agent_prompt.lower()

    assert "only when the user is asking about those private assistant internals" in prompt
    assert "sticky only for re-asks of the same declined request" in prompt
    assert "a new, unrelated question" in prompt
    assert "must be researched and answered later" in prompt
