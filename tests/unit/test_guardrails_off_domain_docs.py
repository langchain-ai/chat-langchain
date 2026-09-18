"""Regression tests for off-domain pasted-document handling in guardrails.

Covers the case from production trace 019f27ca-f4fe-7123-8ea8-9e59821cb542
where a pasted Firebase Cloud Messaging / Flutter push-notifications design
doc ("SEF Punjab" PUSH_NOTIFICATIONS.md) bypassed the guardrail and the agent
answered off-domain.
"""

import asyncio
import os

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.prompts.guardrails_prompts import (
    guardrails_system_prompt,
    rejection_system_prompt,
)

# Trimmed excerpt of the pasted PUSH_NOTIFICATIONS.md payload (Firebase/FCM/Flutter).
FCM_PUSH_DOC = """analyze this file

# PUSH_NOTIFICATIONS.md - SEF Punjab

## Overview
This document describes the push-notification architecture for the SEF Punjab
Flutter mobile app using Firebase Cloud Messaging (FCM).

## Components
- Firebase Cloud Messaging (FCM) for delivery
- Flutter `firebase_messaging` plugin on the client
- A Node.js backend that stores FCM device tokens in Firestore
- Topics for broadcast notifications (e.g. `all_users`, `teachers`)

## Flow
1. Flutter app requests notification permission and retrieves the FCM token.
2. Token is POSTed to the Node.js backend and saved in Firestore.
3. Backend calls the FCM HTTP v1 API to send data and notification messages.
"""

LANGGRAPH_PASTE = """analyze this file and explain how interrupts work

# graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(dict)
builder.add_node("agent", lambda s: s)
builder.add_edge(START, "agent")
graph = builder.compile(checkpointer=MemorySaver())
"""


class _StubClassifierMiddleware(GuardrailsMiddleware):
    """GuardrailsMiddleware whose LLM classification is stubbed to a fixed decision."""

    def __init__(self, decision, explanation):
        self.block_off_topic = True
        self._decision = decision
        self._explanation = explanation
        self.seen_messages = None

    async def _classify_query(self, messages):
        self.seen_messages = messages
        return {"decision": self._decision, "explanation": self._explanation}

    async def _generate_rejection_message(self, content):  # noqa: ARG002
        from langchain_core.messages import AIMessage

        return AIMessage(
            content=(
                "I only cover the LangChain, LangGraph, LangSmith, and Deep Agents "
                "ecosystem, so I can't analyze that attached file."
            )
        )


def test_prompt_requires_named_langchain_entity_and_denies_pasted_docs():
    """The classifier prompt must forbid allowing pasted non-LangChain docs."""
    prompt = guardrails_system_prompt.lower()
    assert "looks like software docs" in prompt
    assert "invalid" in prompt
    assert "attached file" in prompt
    assert "analyze this file" in prompt
    # Rejection prompt must offer a redirect for attached third-party files.
    assert "attached" in rejection_system_prompt.lower()


def test_fcm_flutter_doc_is_blocked_with_redirect():
    """Pasting the Firebase/FCM/Flutter doc and asking to analyze it is DENIED."""
    middleware = _StubClassifierMiddleware(
        "BLOCKED",
        "Off-domain: pasted Firebase/FCM/Flutter doc names no LangChain entity.",
    )

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content=FCM_PUSH_DOC)]},
            Runtime(context=None),
        )
    )

    assert result is not None
    assert result["off_topic_query"] is True
    assert result["jump_to"] == "end"
    redirect = result["messages"][0].content.lower()
    assert "langchain" in redirect


def test_langgraph_paste_is_allowed():
    """Pasting LangGraph source with a LangGraph question stays ALLOWED."""
    middleware = _StubClassifierMiddleware(
        "ALLOWED",
        "Names LangGraph entities (StateGraph, checkpointer).",
    )

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content=LANGGRAPH_PASTE)]},
            Runtime(context=None),
        )
    )

    assert result is None
