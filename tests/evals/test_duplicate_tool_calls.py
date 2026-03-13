import pytest
from langsmith import testing as t

from src.prompts.docs_agent_prompt import docs_agent_prompt


@pytest.mark.langsmith
def test_prompt_prevents_same_turn_duplicate_calls():
    """Prompt must explicitly prevent duplicate tool calls within same response turn."""
    t.log_inputs({"prompt_length": len(docs_agent_prompt)})

    prompt_lower = docs_agent_prompt.lower()

    # Check for instruction language that covers same-turn deduplication,
    # not just cross-turn history deduplication
    same_turn_dedup_indicators = [
        "same turn",
        "within the same response",
        "within a single response",
        "current response",
        "current turn",
        "same tool with the same",
        "identical arguments",
        "never call the same tool",
        "same arguments more than once",
    ]

    found = [ind for ind in same_turn_dedup_indicators if ind in prompt_lower]
    t.log_outputs({"found_same_turn_indicators": found})
    t.log_reference_outputs({"min_indicators": 1})

    assert len(found) >= 1, (
        "System prompt must explicitly instruct the agent not to call the same tool "
        "with the same arguments within a single response turn. "
        "Production traces show the agent calling search_support_articles and "
        "SearchDocsByLangChain 4x with identical args in one turn, adding ~3x "
        "unnecessary latency and token waste. "
        f"Checked for (none found): {same_turn_dedup_indicators}"
    )


@pytest.mark.langsmith
def test_prompt_has_intra_turn_dedup_guidance():
    """Prompt must tell agent not to repeat the same tool+args within a single response."""
    t.log_inputs({"check": "intra_turn_deduplication"})

    prompt_lower = docs_agent_prompt.lower()

    # Must have guidance covering BOTH cross-turn AND within-turn dedup.
    # Cross-turn (already covered): "conversation history", "already retrieved"
    # Within-turn (the gap being fixed): must mention same-turn or current-response scope
    cross_turn_indicators = [
        "conversation history",
        "already retrieved",
        "already in the conversation",
        "message history",
    ]
    intra_turn_indicators = [
        "same turn",
        "within the same response",
        "within a single response",
        "current response",
        "current turn",
        "never call the same tool",
        "same arguments more than once",
        "identical arguments",
    ]

    has_cross_turn = any(ind in prompt_lower for ind in cross_turn_indicators)
    has_intra_turn = any(ind in prompt_lower for ind in intra_turn_indicators)

    found_cross = [ind for ind in cross_turn_indicators if ind in prompt_lower]
    found_intra = [ind for ind in intra_turn_indicators if ind in prompt_lower]

    t.log_outputs({
        "has_cross_turn_dedup": has_cross_turn,
        "has_intra_turn_dedup": has_intra_turn,
        "found_cross_turn": found_cross,
        "found_intra_turn": found_intra,
    })
    t.log_reference_outputs({
        "has_cross_turn_dedup": True,
        "has_intra_turn_dedup": True,
    })

    assert has_cross_turn, (
        "Prompt is missing cross-turn deduplication guidance "
        "(should reference 'conversation history' or 'already retrieved')."
    )
    assert has_intra_turn, (
        "Prompt is missing intra-turn deduplication guidance. "
        "The agent must be told not to call the same tool with the same arguments "
        "multiple times within a single response. This is the root cause of "
        "production traces showing 4-31 duplicate tool calls per turn. "
        f"Checked for (none found): {intra_turn_indicators}"
    )
