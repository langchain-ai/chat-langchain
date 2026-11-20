"""Tests for the legal and contract question handoff."""

import asyncio

import langsmith as ls
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import docs_agent_middleware
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.legal_handoff_middleware import (
    LEGAL_HANDOFF_MESSAGE,
    SUPPORT_TICKET_ARTICLE_URL,
    LegalHandoffMiddleware,
    is_legal_question,
)


@pytest.mark.parametrize(
    "query",
    [
        # Pylon #32321 / #30996 shapes
        "Does your DPA cover health data for our patients?",
        "Does your DPA cover health data?",
        "Can you sign our SCCs?",
        "can you sign our sccs?",
        "Do you offer a data processing agreement?",
        "Do you offer data processing terms?",
        "Do you offer a business associate addendum?",
        "What is protected health information?",
        "Do you offer a master subscription agreement?",
        "Do you offer a master services agreement?",
        "What indemnification do you offer?",
        "What are your liability caps?",
        "Can our counsel review the order form?",
        "Are our redlines covered?",
        "We sent an amendment to the DPA, does it align with your legal review process?",
        "Can you sign a BAA with us on the Plus plan?",
        "Are SCCs included for GDPR data transfers?",
        "Is LangSmith HIPAA compliant?",
        "Does the data processing addendum cover sub-processors?",
        "We need standard contractual clauses for our vendor review",
        "Can our lawyer redline the MSA?",
        "What is the limitation of liability in our order form?",
        "Will you accept our contract amendments?",
        "Can we negotiate the agreement terms?",
    ],
)
def test_legal_questions_are_detected(query):
    assert is_legal_question(query)
    result = LegalHandoffMiddleware().before_agent(
        {"messages": [HumanMessage(content=query)]}, runtime=None
    )
    assert result["jump_to"] == "end"
    assert result["legal_handoff"] is True
    assert result["messages"] == [AIMessage(content=LEGAL_HANDOFF_MESSAGE)]


@pytest.mark.parametrize(
    "query",
    [
        "How do I detect cycles in a graph using strongly connected components (SCCs)?",
        "How do I collect customer order forms as JSON in LangGraph and pause with interrupt()?",
        "How do I process health data in a LangGraph node?",
        "How do I parse DPAs, BAAs, and MSAs in a document pipeline?",
        "How do I display redlines in a text editor?",
        "How do I parse SCCs in a legally_named field?",
        "How do I define an API contract for my LangGraph tool?",
        "What is a data contract in LangSmith datasets?",
        "How do I sign the webhook request payload?",
        "How do I add a health check to my deployment?",
        "How much does the Plus plan cost per seat?",
        "How do I set data retention for traces?",
        "How do I train a model with Smithtune?",
        "How do I stop a Smithtune training job?",
        "How do I disable LangSmith tracing?",
        "How do I opt out of telemetry with LANGGRAPH_CLI_NO_ANALYTICS?",
        "How do I disable analytics while training a model with Smithtune?",
        "How do I stop a Smithtune training job on my dataset?",
        "How do I stop training jobs stored in a database?",
    ],
)
def test_technical_questions_are_not_detected(query):
    assert not is_legal_question(query)
    for content in [query, [{"type": "text", "text": query}]]:
        assert (
            LegalHandoffMiddleware().before_agent(
                {"messages": [HumanMessage(content=content)]}, runtime=None
            )
            is None
        )


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
                    {"type": "text", "text": "Can you sign this BAA?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
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


def test_handoff_points_to_ticket_path_without_promising_follow_up():
    # Portal chats are unmonitored, so a follow-up only happens once the user
    # opens a ticket themselves.
    assert "Contact support" in LEGAL_HANDOFF_MESSAGE
    assert SUPPORT_TICKET_ARTICLE_URL in LEGAL_HANDOFF_MESSAGE
    assert "follow up" not in LEGAL_HANDOFF_MESSAGE.lower()


@pytest.mark.parametrize(
    "query, term", [("Can you sign our SCCs?", "SCCs"), ("HIPAA", "HIPAA")]
)
def test_handoff_records_triggering_term(query, term):
    with ls.tracing_context(enabled="local"), ls.trace("legal-handoff-test") as run:
        LegalHandoffMiddleware().before_agent(
            {"messages": [HumanMessage(content=query)]}, runtime=None
        )
        assert run.metadata["legal_handoff"] is True
        assert run.metadata["legal_handoff_term"] == term


def test_handoff_only_uses_latest_human_message():
    state = {
        "messages": [
            HumanMessage(content="Can you sign our SCCs?"),
            AIMessage(content=LEGAL_HANDOFF_MESSAGE),
            HumanMessage(content="How do I compute SCCs in a graph?"),
        ]
    }
    assert LegalHandoffMiddleware().before_agent(state, runtime=None) is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "How to opt-out of training dat",
        "How to opt-out of training dat  ",
        "No how do I opt out of langsmith training on our traces",
        "how do I opt out of langsmith training on our traces",
        "How can I keep my data out of model training?",
        "Stop using our prompts for training models",
        "How can I prevent customer data from being used to train models?",
        "I don't want my data used for model training",
        "How do we exclude our conversations from fine-tuning?",
        "How do I opt out of training-related retention of customer traces?",
        "Is disabling LANGSMITH_TRACING how I opt out of training on my data?",
    ],
)
def test_training_opt_out_ends_run_before_model(query, asynchronous):
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="Unsupported policy or telemetry instructions")]
    )
    graph = create_agent(model=model, middleware=[LegalHandoffMiddleware()])
    state = {"messages": [HumanMessage(content=query)]}

    result = asyncio.run(graph.ainvoke(state)) if asynchronous else graph.invoke(state)

    answer = result["messages"][-1].content
    assert result["legal_handoff"] is True
    assert "can't confirm" in answer
    assert "Contact support" in answer
    assert SUPPORT_TICKET_ARTICLE_URL in answer
    assert "LANGSMITH_TRACING" not in answer
    assert "LANGGRAPH_CLI_NO_ANALYTICS" not in answer
    assert "Unsupported policy" not in answer


def test_policy_question_with_tool_evidence_remains_on_model_path():
    state = {
        "messages": [
            HumanMessage(content="Does LangSmith use customer traces to train models?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {}, "id": "policy"}],
            ),
            ToolMessage(
                content="Policy: customer traces are not used to train models. Source: policy page.",
                tool_call_id="policy",
            ),
        ]
    }

    assert LegalHandoffMiddleware().before_agent(state, runtime=None) is None


def test_training_handoff_does_not_capture_later_telemetry_question():
    state = {
        "messages": [
            HumanMessage(content="How do I opt out of training on our traces?"),
            AIMessage(content="Please open a support ticket."),
            HumanMessage(content="How do I disable telemetry instead?"),
        ]
    }

    assert LegalHandoffMiddleware().before_agent(state, runtime=None) is None
