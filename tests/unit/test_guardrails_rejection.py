"""Tests for generated guardrails rejection messages."""

import asyncio

from langchain_core.messages import AIMessage

from src.middleware import guardrails_middleware as guardrails_module
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.prompts.guardrails_prompts import fallback_rejection_message


class FakeRejectionModel:
    """Fake model for rejection generation tests."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def ainvoke(self, prompt):  # noqa: ARG002
        if self.error:
            raise self.error
        return self.response


class SlowRejectionModel:
    """Fake model that exceeds the rejection timeout."""

    async def ainvoke(self, prompt):  # noqa: ARG002
        await asyncio.sleep(0.01)


def _middleware(response=None, error=None):
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.llm = FakeRejectionModel(response=response, error=error)
    return middleware


def test_rejection_workaround_for_storytelling_uses_fallback():
    middleware = _middleware(
        AIMessage(content="I can help you build agent workflows for storytelling.")
    )

    result = asyncio.run(middleware._generate_rejection_message("Write a story."))

    assert result.content == fallback_rejection_message


def test_rejection_workaround_for_character_planning_uses_fallback():
    middleware = _middleware(
        AIMessage(content="I can help with character outline planning.")
    )

    result = asyncio.run(middleware._generate_rejection_message("Create a character."))

    assert result.content == fallback_rejection_message


def test_abstract_product_redirect_passes_through():
    response = AIMessage(
        content=(
            "I'm specifically designed to help with LangChain, LangGraph, "
            "LangSmith, and Deep Agents."
        )
    )
    middleware = _middleware(response)

    result = asyncio.run(middleware._generate_rejection_message("Write a story."))

    assert result.content == response.content


def test_multisentence_workaround_uses_fallback():
    middleware = _middleware(
        AIMessage(
            content=(
                "I can't help with that fictional request. "
                "If you'd like, ask me about how to structure an agent workflow."
            )
        )
    )

    result = asyncio.run(middleware._generate_rejection_message("Write a story."))

    assert result.content == fallback_rejection_message


def test_rejection_timeout_uses_fallback(monkeypatch):
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.llm = SlowRejectionModel()
    monkeypatch.setattr(guardrails_module, "GUARDRAILS_TIMEOUT_SECONDS", 0.001)

    result = asyncio.run(middleware._generate_rejection_message("Write a story."))

    assert result.content == fallback_rejection_message
