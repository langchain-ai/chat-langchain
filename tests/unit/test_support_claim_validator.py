from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.support_claim_validator import (
    SUPPORT_PORTAL_REDIRECT,
    SupportClaimValidatorMiddleware,
    sanitize_support_action_claims,
)


def test_sanitizes_flagging_claim():
    content = "I have flagged this for our support team to expedite your name update."

    assert sanitize_support_action_claims(content) == SUPPORT_PORTAL_REDIRECT


def test_sanitizes_escalation_claim():
    content = "I have escalated your request directly to the support operations queue."

    assert sanitize_support_action_claims(content) == SUPPORT_PORTAL_REDIRECT


def test_after_agent_replaces_final_ai_message():
    message = AIMessage(
        content="I have flagged this for our support team to expedite your name update.",
        id="answer-1",
    )
    state = {"messages": [HumanMessage(content="Help", id="question-1"), message]}

    update = SupportClaimValidatorMiddleware().after_agent(
        state, runtime=SimpleNamespace()
    )

    assert update is not None
    assert update["messages"][0].id == "answer-1"
    assert update["messages"][0].content == SUPPORT_PORTAL_REDIRECT
