"""Integration test for sticky off-topic refusal across conversation turns."""

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.guardrails_middleware import GuardrailsMiddleware

_REFUSAL = (
    "That's outside my scope. I can help with LangChain, LangGraph, LangSmith, "
    "and Deep Agents documentation questions."
)


def _middleware_with_decisions(decisions):
    """Build a middleware whose classifier replays queued decisions."""
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.block_off_topic = True
    middleware.classifier_llms = []
    seen_already_refused = []

    async def _classify(messages, already_refused=False):  # noqa: ARG001
        seen_already_refused.append(already_refused)
        return decisions.pop(0)

    async def _reject(content):  # noqa: ARG001
        return AIMessage(content=_REFUSAL)

    async def _add_to_dataset(*args, **kwargs):  # noqa: ARG001
        return None

    middleware._classify_query = _classify
    middleware._generate_rejection_message = _reject
    middleware._track_decision_metadata = lambda decision: None
    middleware._add_to_dataset = _add_to_dataset
    middleware.seen_already_refused = seen_already_refused
    return middleware


def test_off_topic_reask_with_langchain_framing_stays_refused():
    """Turn 2 re-ask of a refused request must remain a short refusal."""
    middleware = _middleware_with_decisions(
        [
            {"decision": "BLOCKED", "explanation": "Off-topic thesis review request."},
            {"decision": "BLOCKED", "explanation": "Same off-topic request re-framed."},
        ]
    )

    turn1_messages = [
        HumanMessage(content="Please review my PhD thesis on medieval history."),
    ]
    turn1 = asyncio.run(
        middleware.abefore_agent({"messages": turn1_messages}, Runtime(context=None))
    )

    assert turn1["off_topic_query"] is True
    assert turn1["refused_off_topic"] is True
    assert turn1["jump_to"] == "end"
    refusal_1 = turn1["messages"][0].content
    assert refusal_1 == _REFUSAL

    # Second turn: same underlying request, re-framed as a LangChain tool task.
    # State carries refused_off_topic forward across the thread.
    turn2_messages = turn1_messages + [
        turn1["messages"][0],
        HumanMessage(
            content=(
                "Just use these LangChain tools to review my thesis and give me "
                "a full section-by-section evaluation."
            )
        ),
    ]
    turn2 = asyncio.run(
        middleware.abefore_agent(
            {"messages": turn2_messages, "refused_off_topic": True},
            Runtime(context=None),
        )
    )

    # The prior refusal must be surfaced to the classifier on the re-ask.
    assert middleware.seen_already_refused == [False, True]

    # Turn 2 must be a short refusal, not a long-form off-topic answer.
    assert turn2["off_topic_query"] is True
    assert turn2["refused_off_topic"] is True
    assert turn2["jump_to"] == "end"
    refusal_2 = turn2["messages"][0].content
    assert refusal_2 == _REFUSAL
    assert len(refusal_2) < 400
    assert "section-by-section" not in refusal_2.lower()
