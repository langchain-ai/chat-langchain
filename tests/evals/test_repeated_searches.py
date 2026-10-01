from pathlib import Path

import pytest
from langsmith import testing as t

docs_agent_prompt = (Path(__file__).parents[2] / "instructions.md").read_text()

HARDENING_RULES = (
    "A follow-up question inside an ongoing conversation is NOT a clarification.",
    "Documentation you read on an earlier turn is NOT evidence for a new question.",
    "`check_links` is link validation, not research, and never satisfies this rule.",
    "If your current answer contradicts anything you said earlier in this conversation, re-read the docs before replying and state plainly which of the two is correct.",
    'If either support KB tool returns an error or the support knowledge-base research leg otherwise fails, explicitly say: "Support articles could not be consulted, so this answer is based on official documentation only."',
    "At most once per turn, on the final citation list immediately before finalizing your response",
    "Never revalidate links already reported valid earlier in the turn.",
    "Call `check_links` at most once, on the final citation list",
)


@pytest.mark.parametrize("rule", HARDENING_RULES)
def test_prompt_contains_hardening_rule(rule):
    """Prompt must retain every recovered hardening rule."""
    assert rule in docs_agent_prompt


@pytest.mark.langsmith
def test_prompt_instructs_agent_to_avoid_repeat_searches():
    """Prompt must instruct agent not to repeat searches already in conversation."""
    t.log_inputs({"prompt_length": len(docs_agent_prompt)})

    has_no_repeat_instruction = (
        "Never call `search_docs_by_lang_chain` or `search_support_articles` "
        "with a query that already has results in the message history - "
        "re-searching duplicates context and causes token overflow"
        in docs_agent_prompt
    )

    t.log_outputs({"has_no_repeat_instruction": has_no_repeat_instruction})
    t.log_reference_outputs({"has_no_repeat_instruction": True})

    assert has_no_repeat_instruction, (
        "System prompt must instruct the agent not to repeat searches "
        "for queries already retrieved in the conversation history. "
        "This causes context overflow from duplicate tool results."
    )


@pytest.mark.langsmith
def test_prompt_has_check_history_before_search_guidance():
    """Prompt must tell agent to check conversation history before calling tools."""
    t.log_inputs({"check": "history_before_search"})

    prompt_lower = docs_agent_prompt.lower()

    # The prompt should have SOME guidance about not duplicating searches
    # that are already in context
    indicators = [
        "already searched",
        "previously searched",
        "already retrieved",
        "already in the conversation",
        "history before",
        "before searching",
        "re-search",
        "re-fetching",
        "duplicate search",
        "already have",
    ]

    found = [ind for ind in indicators if ind in prompt_lower]
    t.log_outputs({"found_indicators": found})
    t.log_reference_outputs({"min_indicators": 1})

    assert len(found) >= 1, (
        f"Prompt missing anti-duplication guidance. "
        f"Checked for: {indicators}"
    )
