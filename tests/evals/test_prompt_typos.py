# tests/evals/test_prompt_typos.py
"""Tests that the system prompt has no known typos or contradictions."""
import pytest
from langsmith import testing as t


@pytest.mark.langsmith
def test_no_inscrase_typo():
    """The prompt must not contain the typo 'inscrase' (misspelling of decrease)."""
    from src.prompts.docs_agent_prompt import docs_agent_prompt

    t.log_inputs({"check": "no 'inscrase' typo in prompt"})
    has_typo = "inscrase" in docs_agent_prompt
    t.log_outputs({"has_typo": has_typo})
    t.log_reference_outputs({"expected": False})

    assert not has_typo, (
        "Prompt contains typo 'inscrase' on the page_size line. "
        "This sends confusing instructions to the LLM."
    )


@pytest.mark.langsmith
def test_no_dpeending_typo():
    """The prompt must not contain the typo 'dpeending' (misspelling of depending)."""
    from src.prompts.docs_agent_prompt import docs_agent_prompt

    t.log_inputs({"check": "no 'dpeending' typo in prompt"})
    has_typo = "dpeending" in docs_agent_prompt
    t.log_outputs({"has_typo": has_typo})
    t.log_reference_outputs({"expected": False})

    assert not has_typo, (
        "Prompt contains typo 'dpeending' on the page_size line. "
        "This sends confusing instructions to the LLM."
    )


@pytest.mark.langsmith
def test_page_size_instruction_is_consistent():
    """The page_size instruction must not tell the agent it can go above 5 when max is 5."""
    from src.prompts.docs_agent_prompt import docs_agent_prompt

    t.log_inputs({"check": "page_size instruction consistency"})

    # The problematic text is "increase or inscrase size" without noting the 5-limit
    # After fix, this should not be present
    has_contradiction = "inscrase" in docs_agent_prompt or "your can increase or" in docs_agent_prompt
    t.log_outputs({"has_contradiction": has_contradiction})
    t.log_reference_outputs({"expected": False})

    assert not has_contradiction, (
        "Prompt contains contradictory or malformed page_size instruction. "
        "The instruction on the page_size default line must be clear and correct."
    )
