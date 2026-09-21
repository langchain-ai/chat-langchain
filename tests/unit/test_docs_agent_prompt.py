"""Tests for docs agent prompt requirements."""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_docs_agent_prompt_requires_language_specific_identifier_naming():
    assert (
        "Method names, attribute names, and keyword-argument names inside a ```python "
        "fence must use the Python snake_case form"
    ) in docs_agent_prompt
    assert "```typescript or ```javascript fence must use the JavaScript/TypeScript camelCase form" in docs_agent_prompt
    assert "Preserve documented class names, imported symbols, constants" in docs_agent_prompt
