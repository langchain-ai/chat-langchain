"""Tests for guardrail verdict state and prompt assembly."""

import asyncio
import os

os.environ["USE_LOCAL_PROMPTS"] = "1"

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from src.middleware.guardrails_middleware import GuardrailsMiddleware


def _middleware_with_decision(decision: str) -> GuardrailsMiddleware:
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.block_off_topic = True

    async def _classify_query(messages, guardrail_history=None):  # noqa: ARG001
        return {"decision": decision, "explanation": "test decision"}

    middleware._classify_query = _classify_query
    middleware._track_decision_metadata = lambda decision: None
    return middleware


def test_allowed_verdict_is_recorded_and_clears_stale_flag():
    middleware = _middleware_with_decision("ALLOWED")

    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [HumanMessage(content="How do I use LangGraph?")],
                "off_topic_query": True,
            },
            Runtime(context=None),
        )
    )

    assert result["scope_verdict"] == "ALLOWED"
    assert result["off_topic_query"] is False


def test_blocked_verdict_is_recorded():
    middleware = _middleware_with_decision("BLOCKED")
    async def _rejection_message(content):  # noqa: ARG001
        return AIMessage(content="Not in scope")

    middleware._generate_rejection_message = _rejection_message

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="What is the weather?")]},
            Runtime(context=None),
        )
    )

    assert result["scope_verdict"] == "BLOCKED"
    assert result["off_topic_query"] is True


def test_allowed_verdict_adds_in_scope_instruction_to_model_prompt():
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="answer")])

    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="How do I use LangGraph?")],
        system_message=SystemMessage(content="Base system prompt"),
        state={"scope_verdict": "ALLOWED"},
    )
    asyncio.run(middleware.awrap_model_call(request, handler))

    assert "classified it IN SCOPE" in calls[0].system_prompt
    assert "Do not answer with a scope refusal" in calls[0].system_prompt


def test_non_allowed_verdict_does_not_add_in_scope_instruction():
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="answer")])

    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="What is the weather?")],
        system_message=SystemMessage(content="Base system prompt"),
        state={"scope_verdict": "BLOCKED"},
    )
    asyncio.run(middleware.awrap_model_call(request, handler))

    assert calls[0].system_prompt == "Base system prompt"
