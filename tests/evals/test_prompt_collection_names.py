# tests/evals/test_prompt_collection_names.py
"""Tests that collection names in the system prompt match those in the tool docstring."""
import re
import pytest
from langsmith import testing as t


def _extract_collection_names_from_text(text: str) -> set[str]:
    """Extract collection names from a bullet list like '- "Name" - description'."""
    # Match quoted collection names in bullet lists
    pattern = r'[-*]\s+"([^"]+)"'
    return set(re.findall(pattern, text))


@pytest.mark.langsmith
def test_prompt_oss_collection_name_matches_tool():
    """The OSS collection name in the prompt must match the tool docstring.
    
    The prompt previously had 'OSS' but the tool docstring (which reflects the actual
    Pylon API collection name) has 'OSS (LangChain and LangGraph)'.
    """
    from src.prompts.docs_agent_prompt import docs_agent_prompt
    from src.tools.pylon_tools import search_support_articles

    t.log_inputs({"check": "OSS collection name consistency between prompt and tool"})

    # Get collection names from prompt
    prompt_collections = _extract_collection_names_from_text(docs_agent_prompt)
    
    # Get collection names from tool docstring
    tool_collections = _extract_collection_names_from_text(search_support_articles.__doc__ or "")

    t.log_outputs({
        "prompt_collections": sorted(prompt_collections),
        "tool_collections": sorted(tool_collections),
    })
    t.log_reference_outputs({"expected": "All tool collection names present in prompt"})

    # The tool has authoritative names (from API); prompt should match
    missing = tool_collections - prompt_collections
    assert not missing, (
        f"These collection names are in the tool docstring but NOT in the prompt: {missing}. "
        f"The prompt has: {prompt_collections}"
    )


@pytest.mark.langsmith
def test_prompt_does_not_have_abbreviated_oss_name():
    """The prompt must not use the abbreviated 'OSS' name that doesn't match the API."""
    from src.prompts.docs_agent_prompt import docs_agent_prompt
    from src.tools.pylon_tools import search_support_articles

    t.log_inputs({"check": "no abbreviated 'OSS' collection name in prompt"})

    # Check that 'OSS (LangChain and LangGraph)' is present in prompt
    has_full_name = "OSS (LangChain and LangGraph)" in docs_agent_prompt
    
    t.log_outputs({"has_full_oss_name": has_full_name})
    t.log_reference_outputs({"expected": True})

    assert has_full_name, (
        "Prompt must use full collection name 'OSS (LangChain and LangGraph)' "
        "to match the actual Pylon API collection name. "
        "Using abbreviated 'OSS' causes collection filter to return no results."
    )
