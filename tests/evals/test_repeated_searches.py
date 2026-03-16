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


@pytest.mark.langsmith
def test_prompt_has_explicit_tool_call_limit():
    """Prompt must contain an explicit hard limit on total tool calls per turn.

    Traces show 10-31 SearchDocsByLangChain calls per single response cycle.
    The prompt must explicitly cap total tool calls (e.g., at most 6 per turn)
    to prevent runaway search loops.

    Failing traces:
      - 019cf560-09db: 20 SearchDocsByLangChain calls for simple Python question
      - 019cf55d-3c0f: 23 tool calls for 'text splitters' question
      - 019cf560-ced0: 29 tool calls for memory question with duplicate queries
    """
    t.log_inputs({"check": "explicit_tool_call_limit"})

    prompt_lower = docs_agent_prompt.lower()

    # The prompt should have an explicit numeric cap on tool calls per turn
    indicators = [
        "at most 6",
        "maximum 6",
        "no more than 6",
        "limit: use at most 6",
        "hard limit",
        "6 total tool",
        "6 tool calls",
    ]

    found = [ind for ind in indicators if ind in prompt_lower]
    t.log_outputs({"found_indicators": found})
    t.log_reference_outputs({"min_indicators": 1})

    assert len(found) >= 1, (
        "Prompt must contain an explicit hard limit on total tool calls per turn "
        "(e.g., 'at most 6 total tool calls'). Without this, the agent issues "
        "10-31 redundant SearchDocsByLangChain calls per turn.\n\n"
        "Fix: add a HARD LIMIT line near the top of docs_agent_prompt, e.g.:\n"
        "  '**HARD LIMIT: Use at most 6 total tool calls per turn.**'"
    )


@pytest.mark.langsmith
def test_prompt_instructs_agent_to_stop_after_sufficient_context():
    """Prompt must tell agent to stop searching once it has sufficient context.

    75% of traces have 6+ tool calls because the agent keeps searching even
    after it already has the relevant documentation. The prompt must explicitly
    instruct the agent to stop and write the response once it has enough context.

    Failing traces:
      - 019cf560-09db: continues searching after retrieving relevant results
      - 019cf55d-3c0f: redundant follow-up searches for already-covered concepts
    """
    t.log_inputs({"check": "stop_after_sufficient_context"})

    prompt_lower = docs_agent_prompt.lower()

    indicators = [
        "stop searching",
        "stop search",
        "write your response",
        "write the response",
        "sufficient context",
        "already contains",
        "already covered",
        "immediately",
        "do not issue follow",
        "don't issue follow",
    ]

    found = [ind for ind in indicators if ind in prompt_lower]
    t.log_outputs({"found_indicators": found, "count": len(found)})
    t.log_reference_outputs({"min_indicators": 2})

    assert len(found) >= 2, (
        f"Prompt should explicitly instruct the agent to stop searching once "
        f"sufficient context is retrieved (found only {len(found)} indicators). "
        f"Checked for: {indicators}\n\n"
        f"Fix: add language like 'Once you have retrieved documentation results "
        f"covering the user's question, STOP searching and write your response "
        f"immediately. Do NOT issue follow-up searches if the first round already "
        f"contains the relevant information.'"
    )
