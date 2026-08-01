# tests/evals/test_grounded_yes_no_answers.py
"""Regression tests for ungrounded bolded Yes/No architecture answers.

Trace evidence: the same user asked "does langchain use langgraph under the
hood" four times in nine minutes over byte-identical retrieval and received
directly contradictory bolded Yes and No openings. Two prompt rules caused it:
the 1-2 word query cap collapsed the relational question to a single noun, and
writing rule 1 demanded a bolded categorical opening regardless of whether any
retrieved passage supported it.
"""

import pytest
from langsmith import testing as t

from src.prompts.docs_agent_prompt import docs_agent_prompt

REGRESSION_INPUT = "does langchain use langgraph under the hood"
CONCEPTS_PRODUCTS_PAGE = "/oss/python/concepts/products"


@pytest.mark.langsmith
def test_prompt_carves_relational_exception_into_query_rules():
    """Relational questions must be exempt from the 1-2 word query cap."""
    t.log_inputs({"question": REGRESSION_INPUT})

    prompt_lower = docs_agent_prompt.lower()

    has_exception = "exception" in prompt_lower and any([
        "relational" in prompt_lower,
        "under the hood" in prompt_lower,
        "built on" in prompt_lower,
    ])
    names_both_entities = "langchain langgraph" in prompt_lower
    names_products_page = CONCEPTS_PRODUCTS_PAGE in docs_agent_prompt

    t.log_outputs({
        "has_exception": has_exception,
        "names_both_entities": names_both_entities,
        "names_products_page": names_products_page,
    })
    t.log_reference_outputs({
        "has_exception": True,
        "names_both_entities": True,
        "names_products_page": True,
    })

    assert has_exception, (
        "Query extraction rules must carve an explicit EXCEPTION for relational "
        f"questions like {REGRESSION_INPUT!r}. The 1-2 word cap collapses them to "
        "single nouns such as 'products' or 'architecture', which retrieve "
        "off-target pages and cannot ground a Yes/No answer."
    )
    assert names_both_entities and names_products_page, (
        "The relational exception must tell the agent to query a phrase naming "
        "BOTH entities, or to read the concepts page directly "
        f"({CONCEPTS_PRODUCTS_PAGE}), which is the page that settles this question."
    )


@pytest.mark.langsmith
def test_prompt_forbids_unsupported_categorical_opening():
    """Bolded Yes/No openings must require a passage retrieved in the same turn."""
    t.log_inputs({"question": REGRESSION_INPUT})

    prompt_lower = docs_agent_prompt.lower()

    forbids_bare_yes_no = "yes/no" in prompt_lower and any([
        "never open with a categorical" in prompt_lower,
        "never manufacture" in prompt_lower,
        "unless a passage" in prompt_lower,
    ])
    excludes_non_passages = all([
        "title" in prompt_lower,
        "check_links" in prompt_lower,
        "not passages" in prompt_lower or "are not passages" in prompt_lower,
    ])
    excludes_tool_errors = "nonetype" in prompt_lower or "429" in prompt_lower
    has_fallback_opening = "what the docs do say" in prompt_lower

    t.log_outputs({
        "forbids_bare_yes_no": forbids_bare_yes_no,
        "excludes_non_passages": excludes_non_passages,
        "excludes_tool_errors": excludes_tool_errors,
        "has_fallback_opening": has_fallback_opening,
    })
    t.log_reference_outputs({
        "forbids_bare_yes_no": True,
        "excludes_non_passages": True,
        "excludes_tool_errors": True,
        "has_fallback_opening": True,
    })

    assert forbids_bare_yes_no, (
        "Writing rule 1 must forbid opening with a categorical Yes/No unless a "
        "passage retrieved in the current turn states the fact. Without this, "
        f"{REGRESSION_INPUT!r} produces contradictory Yes and No answers across "
        "runs over identical retrieval."
    )
    assert excludes_non_passages, (
        "Writing rule 1 must state that page titles, URLs, and check_links "
        "results are not passages and cannot support a Yes/No answer."
    )
    assert excludes_tool_errors, (
        "Writing rule 1 must state that tool error envelopes (NoneType, HTTP 429) "
        "provide no support for either polarity."
    )
    assert has_fallback_opening, (
        "Writing rule 1 must give the agent a grounded fallback: state what the "
        "docs DO say and name the page read, instead of inventing a Yes/No."
    )


@pytest.mark.langsmith
def test_prompt_requires_reading_viewed_page_before_answering():
    """Reading the currently-viewed page must be a hard, ordered precondition."""
    t.log_inputs({"check": "read_viewed_page_precondition"})

    prompt_lower = docs_agent_prompt.lower()

    is_hard_precondition = "hard precondition" in prompt_lower
    gates_answer_composition = any([
        "before you write" in prompt_lower,
        "before composing" in prompt_lower,
    ])
    search_hit_insufficient = "not satisfied by" in prompt_lower

    t.log_outputs({
        "is_hard_precondition": is_hard_precondition,
        "gates_answer_composition": gates_answer_composition,
        "search_hit_insufficient": search_hit_insufficient,
    })
    t.log_reference_outputs({
        "is_hard_precondition": True,
        "gates_answer_composition": True,
        "search_hit_insufficient": True,
    })

    assert is_hard_precondition and gates_answer_composition, (
        "Reading the page the user is viewing must be a hard precondition that "
        "gates answer composition, not an inline aside. The only run that "
        f"answered {REGRESSION_INPUT!r} correctly was the one that read "
        f"{CONCEPTS_PRODUCTS_PAGE}.mdx."
    )
    assert search_hit_insufficient, (
        "The precondition must state that a search hit on the same URL does not "
        "satisfy the read requirement."
    )
