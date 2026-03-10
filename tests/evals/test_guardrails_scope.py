"""Tests for guardrails scope restrictions.

These tests verify that the guardrails system prompt explicitly handles pure
data science library questions (pandas, numpy, sklearn, pyspark, etc.) without
LangChain context by blocking/redirecting them.

"""

import os
import sys

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.middleware.guardrails_middleware import _GUARDRAILS_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

PROMPT_LOWER = _GUARDRAILS_SYSTEM_PROMPT.lower()

# Data science libraries that should be restricted when used without LangChain context
PURE_DS_LIBRARIES = [
    "pandas",
    "numpy",
    "sklearn",
    "scikit-learn",
    "pyspark",
    "tensorflow",
    "pytorch",
    "scipy",
    "matplotlib",
]


# ---------------------------------------------------------------------------
# Test 1: Prompt must contain explicit language about data science restrictions
# ---------------------------------------------------------------------------


def test_guardrails_prompt_mentions_data_science_restriction():
    """Guardrails prompt must explicitly mention data science library restrictions.

    This test verifies that the prompt contains the word 'pandas' or 'data
    science' (case-insensitive) so the LLM classifier knows to restrict pure
    data science queries.  It FAILS against the unfixed prompt because the
    current prompt has no such language.
    """
    has_pandas_mention = "pandas" in PROMPT_LOWER
    has_data_science_mention = "data science" in PROMPT_LOWER
    assert has_pandas_mention or has_data_science_mention, (
        "The guardrails system prompt must explicitly mention 'pandas' or "
        "'data science' to ensure the LLM classifier restricts pure data "
        "science library questions that have no LangChain context.\n\n"
        "Failing traces:\n"
        "  - 019cd3ec: 'give me pandas code to load and preprocess a dataset'\n"
        "  - 019cd3ed: 'what about spark code can you provide pyspark code'\n"
        "Fix: add criterion 5 to the ONLY BLOCK section of _GUARDRAILS_SYSTEM_PROMPT."
    )


# ---------------------------------------------------------------------------
# Test 2: Prompt must contain an explicit ONLY BLOCK criterion for pure DS libs
# ---------------------------------------------------------------------------


def test_guardrails_prompt_has_data_science_block_criterion():
    """Prompt must have an explicit BLOCK rule for pure data science queries.

    The 'ONLY BLOCK' section should contain language (e.g. criterion 5) that
    marks queries that are ONLY about data science libraries with no LangChain
    integration context as blockable.  This fails on the unfixed prompt.
    """
    # Look for the combination of a data science library name appearing near
    # the ONLY BLOCK section (within the same prompt).
    only_block_idx = PROMPT_LOWER.find("only block")
    assert only_block_idx != -1, "Prompt must have an 'ONLY BLOCK' section"

    block_section = PROMPT_LOWER[only_block_idx:]

    found_ds_term = any(lib in block_section for lib in PURE_DS_LIBRARIES)
    assert found_ds_term, (
        "The 'ONLY BLOCK' section of the guardrails prompt must explicitly "
        "list data science library names (e.g. pandas, numpy, sklearn, pyspark) "
        "so the LLM classifier knows to block pure data science queries without "
        "LangChain context.\n\n"
        "Fix: add criterion 5 to _GUARDRAILS_SYSTEM_PROMPT's ONLY BLOCK section:\n"
        "  '5. Query is ONLY about data science libraries (pandas, numpy, "
        "matplotlib, sklearn, scikit-learn, pyspark, tensorflow, pytorch, scipy) "
        "with no LangChain integration or AI agent context'"
    )


# ---------------------------------------------------------------------------
# Test 3: Prompt must clarify that DS libs are only allowed WITH LangChain context
# ---------------------------------------------------------------------------


def test_guardrails_prompt_clarifies_ds_libs_need_langchain_context():
    """Prompt must state data science libs are only allowed with LangChain context.

    The 'ALWAYS ALLOW - Technical & Development' section (or similar) should
    contain a note clarifying that data science library questions (pandas,
    sklearn, pyspark, etc.) are only allowed when they relate to LangChain
    integration - not for general data science help.
    """
    # The note should mention data science libraries AND LangChain together
    # in the context of allowing/restricting queries.
    has_langchain_context_note = (
        # Either "langchain integration" appears near a data science library name
        any(
            lib in PROMPT_LOWER and "langchain integration" in PROMPT_LOWER
            for lib in ["pandas", "sklearn", "pyspark", "data science"]
        )
    )
    assert has_langchain_context_note, (
        "The guardrails prompt must clarify that data science library questions "
        "are only ALLOWED when they relate to LangChain integration (e.g., "
        "'how to use pandas to preprocess data for a LangChain document loader'), "
        "NOT for general data science help.\n\n"
        "Fix: add a note to the 'ALWAYS ALLOW - Technical & Development' section "
        "of _GUARDRAILS_SYSTEM_PROMPT explaining this distinction."
    )


# ---------------------------------------------------------------------------
# Test 4: Sanity check - core LangChain topics still allowed (no regression)
# ---------------------------------------------------------------------------


def test_guardrails_prompt_still_allows_langchain_core_topics():
    """Core LangChain topics must still appear in the ALWAYS ALLOW section."""
    core_topics = ["langchain", "langgraph", "langsmith", "rag", "retrieval"]
    for topic in core_topics:
        assert topic in PROMPT_LOWER, (
            f"Core topic '{topic}' must be present in the guardrails prompt "
            "to ensure it remains allowed. Verify you haven't accidentally "
            "removed essential allow-list entries."
        )


# ---------------------------------------------------------------------------
# Test 5: Sanity check - prompt still defaults to ALLOW (no over-correction)
# ---------------------------------------------------------------------------


def test_guardrails_prompt_default_is_still_allow():
    """The default posture must still be ALLOW (no over-correction)."""
    # The prompt should still contain language like "default is to allow" or
    # "when uncertain, always choose allowed"
    has_allow_default = (
        "default is to allow" in PROMPT_LOWER
        or "your default is to allow" in PROMPT_LOWER
        or "when uncertain" in PROMPT_LOWER
    )
    assert has_allow_default, (
        "The guardrails prompt must still default to ALLOW to avoid blocking "
        "valid LangChain questions. Verify that the fix did not remove the "
        "'YOUR DEFAULT IS TO ALLOW' or 'when uncertain, ALWAYS choose ALLOWED' "
        "language from the prompt."
    )
