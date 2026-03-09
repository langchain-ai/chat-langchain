"""Tests that verify the prompt explicitly limits parallel searches to at most 2 per turn.

Root cause: the prompt said '1-2 searches in parallel' which is ambiguous — in production
traces the agent made 4-5 parallel SearchDocsByLangChain calls per turn, causing 40+ second
latency. The fix ensures the prompt contains unambiguous limiting language.
"""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_prompt_is_string():
    """docs_agent_prompt should be a plain string."""
    assert isinstance(docs_agent_prompt, str), (
        "docs_agent_prompt should be a string, got %s" % type(docs_agent_prompt)
    )


def test_prompt_contains_parallel_search_guidance():
    """The prompt must contain guidance about parallel searches."""
    prompt_lower = docs_agent_prompt.lower()
    assert "parallel" in prompt_lower, (
        "Prompt should mention 'parallel' searches but none found"
    )


def test_prompt_limits_parallel_doc_searches_to_max_2():
    """Prompt should explicitly limit parallel doc searches to max 2 per turn.

    The vague wording '1-2 DIFFERENT page titles' allows the agent to interpret
    the upper bound loosely and make 4-5 parallel calls. The fix adds explicit
    limiting language like 'no more than 2', 'maximum 2', or 'at most 2'.
    """
    prompt_lower = docs_agent_prompt.lower()
    limiting_phrases = [
        "maximum 2",
        "no more than 2",
        "at most 2",
        "max 2",
        "limit of 2",
        "never more than 2",
    ]
    assert any(phrase in prompt_lower for phrase in limiting_phrases), (
        "Prompt should contain explicit max-2 parallel search constraint. "
        "Searched for: %s. "
        "Current prompt uses vague '1-2' wording that the model interprets loosely, "
        "resulting in 4-5 parallel SearchDocsByLangChain calls per turn (40s+ latency). "
        "Add unambiguous limiting language such as 'no more than 2 searches in parallel'."
        % limiting_phrases
    )


def test_prompt_does_not_use_vague_parallel_search_wording_alone():
    """The prompt should not rely solely on '1-2' to constrain parallel searches.

    '1-2' is a range, not an explicit maximum. By itself it leaves the upper bound
    ambiguous. The fix should accompany or replace it with clear limiting language.
    """
    prompt_lower = docs_agent_prompt.lower()
    limiting_phrases = [
        "maximum 2",
        "no more than 2",
        "at most 2",
        "max 2",
        "limit of 2",
        "never more than 2",
    ]
    has_explicit_limit = any(phrase in prompt_lower for phrase in limiting_phrases)
    assert has_explicit_limit, (
        "Prompt relies on vague '1-2' range for parallel doc searches without an "
        "explicit upper bound. This caused 4-5 parallel calls per turn in production. "
        "Replace or supplement with explicit limiting language."
    )


def test_prompt_parallel_doc_search_section_present():
    """The docs search step should include guidance about how many to run in parallel."""
    # Check the key workflow section exists
    assert "For docs" in docs_agent_prompt or "for docs" in docs_agent_prompt.lower(), (
        "Expected a 'For docs' section in the Research Workflow describing parallel search behavior"
    )
