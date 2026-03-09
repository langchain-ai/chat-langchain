# tests/evals/test_default_page_size.py
import pytest
from langsmith import testing as t


@pytest.mark.langsmith
def test_default_page_size_matches_prompt_instruction():
    """DEFAULT_PAGE_SIZE in docs_tools should match the page_size=5 the prompt instructs.

    The system prompt says 'Start with page_size=5', so the DEFAULT_PAGE_SIZE constant
    must be 5 to avoid a mismatch when the agent omits the parameter.
    """
    from src.tools.docs_tools import DEFAULT_PAGE_SIZE

    t.log_inputs({"constant": "DEFAULT_PAGE_SIZE"})
    t.log_outputs({"value": DEFAULT_PAGE_SIZE})
    t.log_reference_outputs({"expected": 5})

    assert DEFAULT_PAGE_SIZE == 5, (
        f"DEFAULT_PAGE_SIZE is {DEFAULT_PAGE_SIZE} but prompt instructs page_size=5. "
        "This causes the agent to get fewer results than intended when not specifying page_size."
    )


@pytest.mark.langsmith
def test_default_page_size_within_valid_range():
    """DEFAULT_PAGE_SIZE must be between 1 and MAX_PAGE_SIZE."""
    from src.tools.docs_tools import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

    t.log_inputs({"constants": "DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE"})
    t.log_outputs({"default": DEFAULT_PAGE_SIZE, "max": MAX_PAGE_SIZE})
    t.log_reference_outputs({"expected": "1 <= DEFAULT_PAGE_SIZE <= MAX_PAGE_SIZE"})

    assert 1 <= DEFAULT_PAGE_SIZE <= MAX_PAGE_SIZE, (
        f"DEFAULT_PAGE_SIZE={DEFAULT_PAGE_SIZE} out of range [1, {MAX_PAGE_SIZE}]"
    )
