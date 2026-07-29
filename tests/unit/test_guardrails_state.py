"""Tests for guardrails per-turn off-topic state tracking."""

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module
from src.middleware.guardrails_middleware import GuardrailsMiddleware


def _middleware() -> GuardrailsMiddleware:
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = []
    middleware.block_off_topic = True
    return middleware


def _patch_side_effects(monkeypatch, middleware):
    monkeypatch.setattr(guardrails_module, "ALLOWED_SAMPLE_RATE", 0)

    async def _no_dataset(query, result, explanation, preview):  # noqa: ARG001
        return None

    async def _rejection(content):  # noqa: ARG001
        return AIMessage(content="I can only help with LangChain questions.")

    monkeypatch.setattr(middleware, "_add_to_dataset", _no_dataset)
    monkeypatch.setattr(middleware, "_generate_rejection_message", _rejection)


def _decision(decision, explanation):
    async def _classify(messages):  # noqa: ARG001
        return {"decision": decision, "explanation": explanation}

    return _classify


def test_allowed_turn_clears_flag_set_by_earlier_blocked_turn(monkeypatch):
    middleware = _middleware()
    _patch_side_effects(monkeypatch, middleware)

    monkeypatch.setattr(
        middleware, "_classify_query", _decision("BLOCKED", "Off-topic request.")
    )
    state = {"messages": [HumanMessage(content="Write me a poem about pirates.")]}
    blocked_update = asyncio.run(
        middleware.abefore_agent(state, Runtime(context=None))
    )

    assert blocked_update["off_topic_query"] is True

    # The flag is checkpointed, so the next turn of the thread starts from it.
    state["off_topic_query"] = blocked_update["off_topic_query"]
    state["messages"].append(HumanMessage(content="How do LangChain agents work?"))

    monkeypatch.setattr(
        middleware, "_classify_query", _decision("ALLOWED", "LangChain question.")
    )
    allowed_update = asyncio.run(
        middleware.abefore_agent(state, Runtime(context=None))
    )

    assert allowed_update is not None
    assert allowed_update["off_topic_query"] is False
    assert "jump_to" not in allowed_update
