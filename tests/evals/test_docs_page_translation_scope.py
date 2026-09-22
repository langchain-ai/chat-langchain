"""Regression tests for translation requests about viewed docs pages."""

import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt

PAGE_CONTEXT = """
[Page context]
URL: https://docs.langchain.com/oss/python/langchain/overview
The user is currently viewing this LangChain documentation page.
"""


@pytest.mark.parametrize("user_message", ["翻译", "translate to chinese"])
def test_viewed_docs_translation_is_not_scope_refused(user_message):
    """Viewed LangChain docs translation requests must remain in scope."""
    prompt_lower = docs_agent_prompt.lower()
    request = f"{user_message}\n{PAGE_CONTEXT}"

    assert "generic language help or translation requests unrelated" in prompt_lower
    assert "summarize, translate, or expand" in prompt_lower
    assert "currently viewing" in prompt_lower
    assert "query_docs_filesystem_docs_by_lang_chain" in prompt_lower
    assert "scope refusal" not in request.lower()
