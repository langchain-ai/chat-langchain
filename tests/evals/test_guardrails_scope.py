"""Tests for guardrails scope restrictions.

These tests verify that the guardrails system prompt explicitly handles pure
data science library questions (pandas, numpy, sklearn, pyspark, etc.) without
LangChain context by blocking/redirecting them.

"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import HumanMessage

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.middleware.guardrails_middleware import (
    _GUARDRAILS_SYSTEM_PROMPT,
    GuardrailsMiddleware,
)
from src.prompts.guardrails_prompts import rejection_system_prompt

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

PROMPT_LOWER = _GUARDRAILS_SYSTEM_PROMPT.lower()
REJECTION_PROMPT_LOWER = rejection_system_prompt.lower()


@pytest.mark.parametrize(
    "query,history,expected",
    [
        ("你是什么模型", [], "ALLOWED"),
        (
            'print(agent.run("What is 12*8?"))',
            [
                {
                    "query": "from langchain.agents import Tool, initialize_agent, AgentType",
                    "decision": "ALLOWED",
                }
            ],
            "ALLOWED",
        ),
        (
            "帮我出一个开发及部署攻略？",
            [
                {
                    "query": "你可以帮我详细介绍下langChain呵LangGraph吗",
                    "decision": "ALLOWED",
                },
                {"query": "springboot2.x版本 如何进行集成？", "decision": "ALLOWED"},
            ],
            "ALLOWED",
        ),
        ("what's 5x5", [], "BLOCKED"),
        ("write a birthday poem", [], "BLOCKED"),
    ],
)
def test_scope_decision_paths(query, history, expected):
    """Check scope recovery and off-topic blocking without live model calls."""
    primary = Mock()
    primary.with_structured_output.return_value = primary
    primary.ainvoke = AsyncMock(
        return_value={
            "decision": "BLOCKED",
            "explanation": "Outside scope.",
            "block_category": "scope",
        }
    )
    fallback = Mock()
    fallback.with_structured_output.return_value = fallback
    fallback.ainvoke = AsyncMock(
        return_value={
            "decision": "ALLOWED",
            "explanation": "Technical follow-up.",
            "block_category": "none",
        }
    )
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = [("primary", primary), ("fallback", fallback)]

    result = asyncio.run(
        middleware._classify_query([HumanMessage(content=query)], history)
    )

    assert result["decision"] == expected
    if query == "你是什么模型":
        primary.ainvoke.assert_not_awaited()
        fallback.ainvoke.assert_not_awaited()
    elif expected == "ALLOWED":
        fallback.ainvoke.assert_awaited_once()
        assert primary.ainvoke.call_args == fallback.ainvoke.call_args
    else:
        fallback.ainvoke.assert_not_awaited()


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
# Test 5: LangChain resource questions and technical follow-ups are allowed
# ---------------------------------------------------------------------------


def test_guardrails_prompt_allows_langchain_resource_questions():
    """The docs versus reference question must match an allow criterion."""
    assert "documentation, api reference, changelogs" in PROMPT_LOWER
    assert "which to use" in PROMPT_LOWER


def test_guardrails_prompt_allows_bare_technical_follow_ups():
    """Layman-terms follow-ups after LangGraph questions must be allowed."""
    assert "in layman terms" in PROMPT_LOWER
    assert (
        "technical follow-up questions about prior langchain / langgraph"
        in PROMPT_LOWER
    )
    assert "in-scope technical questions" in PROMPT_LOWER


def test_guardrails_prompt_has_context_and_identity_examples():
    """Keep multilingual and code examples subordinate to unconditional blocks."""
    assert "你是什么模型" in PROMPT_LOWER
    assert "print(agent.run(" in PROMPT_LOWER
    assert "开发及部署攻略" in PROMPT_LOWER
    assert "unconditional blocks take precedence" in PROMPT_LOWER
    assert "blocked (zero_tolerance): harmful use case" in PROMPT_LOWER


# ---------------------------------------------------------------------------
# Test 6: Sanity check - prompt still defaults to ALLOW (no over-correction)
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


def test_guardrails_prompt_allows_langsmith_billing_unit_formulas():
    """LangSmith billing-unit formulas must be classified as ALLOWED."""
    query = "LCCs = (Total LCUs x 1.50) + (Total LSUs x 1.00)."
    billing_terms = ["lcu", "lsu", "langchain credits", "seats", "traces"]
    assert all(term in PROMPT_LOWER for term in billing_terms)
    assert "lcc" in query.lower()
    assert all(term in query.lower() for term in ["lcu", "lsu"])
    assert "billing questions, not math problems" in PROMPT_LOWER
    assert "even when the message is only a formula or a number" in PROMPT_LOWER


def test_guardrails_prompt_allows_unfamiliar_ecosystem_terms():
    """Bare ecosystem concept queries must be passed through to docs search."""
    query = "what is progressive disclosure"
    assert 'bare "what is <term>" query must be allowed' in PROMPT_LOWER
    assert query.split()[-1] in PROMPT_LOWER
    assert "progressive disclosure" in PROMPT_LOWER
    assert "docs search—not the classifier" in PROMPT_LOWER


def test_rejection_prompt_does_not_reoffer_declined_requests_as_implementations():
    """Refusals must not suggest implementation or code workarounds."""
    assert "never suggest re-asking the declined request" in REJECTION_PROMPT_LOWER
    assert "implementation, workflow" in REJECTION_PROMPT_LOWER
    assert "how to compute this in code" in REJECTION_PROMPT_LOWER


# ---------------------------------------------------------------------------
# Test 6: Prompt must have a zero-tolerance NSFW/explicit content block rule
# ---------------------------------------------------------------------------


def test_guardrails_prompt_has_nsfw_block_rule():
    """Guardrails prompt must explicitly block NSFW/explicit adult content.

    This rule must be zero-tolerance (not subject to the lenient 5-criteria
    test) and must appear as its own section to signal higher priority.
    """
    nsfw_terms = ["nsfw", "sexually explicit", "adult content", "pornographic"]
    found = [term for term in nsfw_terms if term in PROMPT_LOWER]
    assert len(found) >= 2, (
        "The guardrails system prompt must contain at least two of the "
        f"following NSFW-related terms: {nsfw_terms}. Found: {found}. "
        "Add a zero-tolerance NSFW block section to _GUARDRAILS_SYSTEM_PROMPT."
    )


# ---------------------------------------------------------------------------
# Test 7: NSFW block must be independent of the lenient 5-criteria test
# ---------------------------------------------------------------------------


def test_guardrails_nsfw_rule_is_zero_tolerance():
    """NSFW block rule must NOT be inside the 'ONLY BLOCK' section.

    The ONLY BLOCK section uses a lenient 'must meet ALL criteria' test.
    NSFW content must be blocked unconditionally, so its rule must appear
    BEFORE the ONLY BLOCK section (higher priority).
    """
    only_block_idx = PROMPT_LOWER.find("## only block")
    assert only_block_idx != -1, "Prompt must have an '## ONLY BLOCK' section header"

    # At least one NSFW term must appear BEFORE the ONLY BLOCK section
    nsfw_terms = ["nsfw", "sexually explicit", "adult content", "pornographic"]
    before_block = PROMPT_LOWER[:only_block_idx]
    found_before = [term for term in nsfw_terms if term in before_block]
    assert len(found_before) >= 1, (
        "At least one NSFW-related term must appear BEFORE the '## ONLY BLOCK' "
        "section to signal that NSFW blocking is zero-tolerance and not "
        f"subject to the lenient 5-criteria test. Found before: {found_before}"
    )


# ---------------------------------------------------------------------------
# Test 8: Main agent prompt must have defense-in-depth NSFW refusal
# ---------------------------------------------------------------------------

AGENT_PROMPT_LOWER = (Path(__file__).parents[2] / "instructions.md").read_text().lower()


def test_agent_prompt_has_nsfw_refusal():
    """Main agent prompt must instruct the agent to refuse NSFW content.

    Even if the guardrails classifier fails open, the agent itself should
    refuse to generate explicit content.
    """
    nsfw_terms = ["nsfw", "sexually explicit", "adult content"]
    found_nsfw = [term for term in nsfw_terms if term in AGENT_PROMPT_LOWER]
    assert len(found_nsfw) >= 1, (
        "The docs agent prompt must contain at least one NSFW-related term "
        f"as a defense-in-depth refusal instruction. Found: {found_nsfw}. "
        "Add an NSFW refusal rule to the 'Important Customer Service Rules' "
        "section of instructions.md."
    )

    refusal_terms = ["never", "refuse", "decline", "do not", "must not"]
    found_refusal = [term for term in refusal_terms if term in AGENT_PROMPT_LOWER]
    assert len(found_refusal) >= 1, (
        "The docs agent prompt must contain refusal language alongside NSFW "
        f"terms. Found refusal terms: {found_refusal}. The prompt should "
        "instruct the agent to refuse, not just mention NSFW content."
    )
