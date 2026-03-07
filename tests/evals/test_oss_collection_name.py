"""Tests that the docs_agent_prompt uses the correct OSS collection name.

The real Pylon KB collection is named "OSS (LangChain and LangGraph)".
If the prompt lists just "OSS", the case-insensitive lookup fails and
search_support_articles returns an error for all OSS questions.
"""

import pytest
from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_prompt_contains_full_oss_collection_name():
    """Prompt must contain the full collection name 'OSS (LangChain and LangGraph)'."""
    assert "OSS (LangChain and LangGraph)" in docs_agent_prompt, (
        "docs_agent_prompt should list the collection as "
        "'OSS (LangChain and LangGraph)' to match the real Pylon KB collection name. "
        f"Prompt snippet around OSS: {docs_agent_prompt[docs_agent_prompt.find('OSS') - 10 : docs_agent_prompt.find('OSS') + 60]!r}"
    )


def test_prompt_does_not_contain_bare_oss_collection():
    """Prompt must not list a bare '"OSS"' collection name that would fail the lookup."""
    # The bare name would appear as: - "OSS" - LangChain...
    # After the fix it should be: - "OSS (LangChain and LangGraph)" - LangChain...
    import re

    # Look for the pattern: "OSS" followed by something that is NOT " (LangChain"
    # This catches the old broken entry like: - "OSS" - LangChain and LangGraph...
    bare_oss_pattern = re.compile(r'"OSS"(?!\s*\()')
    match = bare_oss_pattern.search(docs_agent_prompt)
    assert match is None, (
        f"docs_agent_prompt contains bare '\"OSS\"' (at position {match.start()}) "
        "which does not match the real Pylon collection name. "
        "Use '\"OSS (LangChain and LangGraph)\"' instead."
    )


def test_prompt_collection_name_matches_tool_docstring():
    """The collection name in the prompt must match what the tool's docstring says."""
    from src.tools.pylon_tools import search_support_articles

    tool_docstring = search_support_articles.__doc__ or ""
    assert "OSS (LangChain and LangGraph)" in tool_docstring, (
        "search_support_articles docstring should list 'OSS (LangChain and LangGraph)'. "
        "This is the canonical source of truth for valid collection names."
    )
    assert "OSS (LangChain and LangGraph)" in docs_agent_prompt, (
        "docs_agent_prompt must use the same collection name as the tool docstring: "
        "'OSS (LangChain and LangGraph)'"
    )
