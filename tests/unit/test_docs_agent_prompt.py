from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_docs_agent_prompt_forbids_in_source_credentials():
    assert "NEVER tell a user to put a real API key" in docs_agent_prompt
