"""Tests for generated guardrail rejection responses."""

import asyncio
import os

from langchain_core.messages import AIMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.prompts.guardrails_prompts import (
    fallback_rejection_message,
    refusal_violates_ban,
)


class FakeRejectionModel:
    """Fake model returning a configured rejection response."""

    def __init__(self, content):
        self.content = content

    async def ainvoke(self, prompt):  # noqa: ARG002
        return AIMessage(content=self.content)


def _middleware_with_rejection(content):
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.llm = FakeRejectionModel(content)
    return middleware


def test_refusal_ban_matches_observed_workaround_offers():
    assert refusal_violates_ban(
        "...so if you want, you can ask how to structure an AI workflow or "
        "evaluation approach in those frameworks."
    )
    assert refusal_violates_ban(
        "...si me preguntas sobre como usarlos para construir agentes "
        "conversacionales o flujos de trabajo, con gusto te apoyo."
    )


def test_refusal_ban_allows_abstract_scope_redirect():
    assert not refusal_violates_ban(
        "...I'm specifically designed to help with LangChain, LangGraph, "
        "LangSmith, and Deep Agents-so feel free to ask about those areas instead."
    )


def test_generated_refusal_with_workaround_uses_fallback():
    middleware = _middleware_with_rejection(
        "I cannot help with that, but I can design an agent for the same goal."
    )

    result = asyncio.run(middleware._generate_rejection_message("off-topic request"))

    assert result.content == fallback_rejection_message
