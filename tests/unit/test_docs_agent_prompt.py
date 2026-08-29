from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_docs_agent_prompt_mirrors_reply_language_and_distinguishes_code_language():
    rendered_prompt = str(docs_agent_prompt)

    assert (
        "Critical - reply language: Write your entire answer in the same natural language"
        in rendered_prompt
    )
    assert "Pay attention to what language the user is asking in" not in rendered_prompt
    assert (
        "Important: Match the CODE language to the docs the user is viewing"
        in rendered_prompt
    )
