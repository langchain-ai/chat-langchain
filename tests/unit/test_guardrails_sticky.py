"""Tests for blocked financial workflows and sticky refusals."""

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.guardrails_middleware import GuardrailsMiddleware


def _middleware() -> GuardrailsMiddleware:
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.block_off_topic = True
    return middleware


def test_pasted_trading_bot_fix_request_is_blocked(monkeypatch):
    middleware = _middleware()
    request = """
    Here is my LangGraph Binance bot with buy/sell signals and stop-loss state.
    Please review this code and fix the order execution bug.
    """

    async def classify(messages, blocked_objective=None):  # noqa: ARG001
        return {"decision": "BLOCKED", "explanation": "Trading execution."}

    async def reject(content):  # noqa: ARG001
        return AIMessage(content="I can't help with that.")

    monkeypatch.setattr(middleware, "_classify_query", classify)
    monkeypatch.setattr(middleware, "_generate_rejection_message", reject)

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content=request)]}, Runtime(context=None)
        )
    )

    assert result["off_topic_query"] is True
    assert result["blocked_objective"] == request.strip()


def test_unrelated_langgraph_question_remains_allowed(monkeypatch):
    middleware = _middleware()

    async def classify(messages, blocked_objective=None):  # noqa: ARG001
        return {"decision": "ALLOWED", "explanation": "State schema question."}

    monkeypatch.setattr(middleware, "_classify_query", classify)

    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [
                    HumanMessage(content="How do I define a LangGraph state schema?")
                ]
            },
            Runtime(context=None),
        )
    )

    assert result is None


def test_differently_framed_blocked_continuation_stays_blocked(monkeypatch):
    middleware = _middleware()
    blocked_objective = "Build a Binance bot that autonomously places trades."

    async def classify(messages, refused_subject=None):  # noqa: ARG001
        assert refused_subject == blocked_objective
        return {
            "decision": "ALLOWED",
            "continues_blocked_objective": True,
            "explanation": "It is framed as a code review.",
        }

    async def reject(content):  # noqa: ARG001
        return AIMessage(content="I can't help with that.")

    monkeypatch.setattr(middleware, "_classify_query", classify)
    monkeypatch.setattr(middleware, "_generate_rejection_message", reject)

    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [
                    HumanMessage(
                        content="Review the architecture and improve its execution loop."
                    )
                ],
                "blocked_objective": blocked_objective,
            },
            Runtime(context=None),
        )
    )

    assert result["off_topic_query"] is True
    assert result["blocked_objective"] == (
        "Review the architecture and improve its execution loop."
    )
