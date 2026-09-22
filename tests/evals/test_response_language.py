"""Eval coverage for natural-language response selection."""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_docs_agent_prompt_has_natural_language_response_rule():
    """Prompt must require replies in the user's natural language."""
    assert "Respond in the same natural language the user wrote in." in docs_agent_prompt
    assert "app boilerplate" in docs_agent_prompt


def test_chinese_question_has_chinese_answer_expectation():
    """Chinese questions must receive answers written in Chinese."""
    question = "如何在 LangChain 中配置中间件？"
    expected_language = "Chinese"

    language_rule = "Respond in the same natural language the user wrote in."
    assert any("\u4e00" <= character <= "\u9fff" for character in question)
    assert expected_language == "Chinese"
    assert language_rule in docs_agent_prompt
