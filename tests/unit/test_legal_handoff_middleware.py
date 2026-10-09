"""Tests for the legal and contract question handoff."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import docs_agent_middleware
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.legal_handoff_middleware import (
    LEGAL_HANDOFF_MESSAGE,
    LegalHandoffMiddleware,
    is_legal_question,
)


@pytest.mark.parametrize(
    "query",
    [
        # Pylon #32321 / #30996 shapes
        "Does your DPA cover health data for our patients?",
        "We sent an amendment to the DPA, does it align with your legal review process?",
        "Can you sign a BAA with us on the Plus plan?",
        "Are SCCs included for EU transfers?",
        "Is LangSmith HIPAA compliant?",
        "Does the data processing addendum cover sub-processors?",
        "We need standard contractual clauses for our vendor review",
        "Can we redline the MSA?",
        "What is the limitation of liability in our order form?",
        "Will you accept our contract amendments?",
        "Can we negotiate the agreement terms?",
    ],
)
def test_legal_questions_are_detected(query):
    assert is_legal_question(query)


@pytest.mark.parametrize(
    "query",
    [
        "How do I define an API contract for my LangGraph tool?",
        "What is a data contract in LangSmith datasets?",
        "How do I sign the webhook request payload?",
        "How do I add a health check to my deployment?",
        "How much does the Plus plan cost per seat?",
        "How do I set data retention for traces?",
    ],
)
def test_technical_questions_are_not_detected(query):
    assert not is_legal_question(query)


def test_handoff_ends_run_with_fixed_message():
    middleware = LegalHandoffMiddleware()
    state = {"messages": [HumanMessage(content="Does the DPA cover health data?")]}

    result = middleware.before_agent(state, runtime=None)

    assert result["jump_to"] == "end"
    assert result["legal_handoff"] is True
    assert result["messages"] == [AIMessage(content=LEGAL_HANDOFF_MESSAGE)]


def test_handoff_reads_text_blocks_in_multimodal_content():
    middleware = LegalHandoffMiddleware()
    state = {
        "messages": [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Is this BAA acceptable?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ]
            )
        ]
    }

    result = middleware.before_agent(state, runtime=None)

    assert result["jump_to"] == "end"


def test_handoff_ignores_technical_questions():
    middleware = LegalHandoffMiddleware()
    state = {"messages": [HumanMessage(content="How do I stream from a subgraph?")]}

    assert middleware.before_agent(state, runtime=None) is None


def test_handoff_runs_before_guardrails():
    types = [type(m) for m in docs_agent_middleware]
    assert types.index(LegalHandoffMiddleware) < types.index(GuardrailsMiddleware)
