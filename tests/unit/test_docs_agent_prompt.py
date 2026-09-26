"""Tests for the docs agent prompt contract."""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_chinese_response_language_rule_preserves_code_fences():
    """Require Chinese prose while preserving code and other literals."""
    prompt = docs_agent_prompt.lower()

    assert "same natural language as the user's most recent question" in prompt
    assert "keep code, identifiers, api names, error strings, and documentation urls" in prompt
    assert "do not switch the answer to english merely because the retrieved documentation is in english" in prompt
    assert "current turn's question, not from the language of the previous turn's answer" in prompt
    assert "requested code example language" in prompt
    assert "all code wrapped in triple backticks" in prompt
