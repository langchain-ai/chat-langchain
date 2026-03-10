import pytest
from langsmith import testing as t
from src.prompts.docs_agent_prompt import docs_agent_prompt


@pytest.mark.langsmith
def test_prompt_instructs_agent_to_avoid_repeat_searches():
    """Prompt must instruct agent not to repeat searches already in conversation."""
    t.log_inputs({"prompt_length": len(docs_agent_prompt)})

    # Check for anti-repetition instruction in the prompt
    prompt_lower = docs_agent_prompt.lower()

    has_no_repeat_instruction = any([
        "already" in prompt_lower and "search" in prompt_lower,
        "do not re-search" in prompt_lower,
        "don't re-search" in prompt_lower,
        "avoid repeat" in prompt_lower,
        "already retrieved" in prompt_lower,
        "already in" in prompt_lower and "history" in prompt_lower,
        "conversation already" in prompt_lower,
        "skip" in prompt_lower and "already" in prompt_lower,
    ])

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
