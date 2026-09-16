"""Tests for guardrails conversation context and follow-up classification."""

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.guardrails_middleware import GuardrailsMiddleware


class PromptCapturingModel:
    """Fake structured model that captures the classifier prompt."""

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, prompt, config=None):  # noqa: ARG002
        self.prompts.append(prompt)
        return self.result


def _middleware_with_model(model):
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = [("fake", model)]
    middleware.block_off_topic = True
    return middleware


def _prompt_content(model):
    return model.prompts[0][1].content


def test_classifier_context_includes_recent_assistant_turns():
    model = PromptCapturingModel(
        {"decision": "ALLOWED", "explanation": "Follow-up context is in scope."}
    )
    middleware = _middleware_with_model(model)

    asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(content="Explain StateGraph."),
                AIMessage(content="StateGraph defines graph state and nodes."),
                HumanMessage(content="What do these two mean?"),
            ]
        )
    )

    content = _prompt_content(model)
    assert "Recent conversation:" in content
    assert "User: Explain StateGraph." in content
    assert "Assistant: StateGraph defines graph state and nodes." in content
    assert "Previous questions" not in content


def test_classifier_context_truncates_user_and_assistant_messages():
    model = PromptCapturingModel(
        {"decision": "ALLOWED", "explanation": "Context was provided."}
    )
    middleware = _middleware_with_model(model)
    user_text = "u" * 400
    assistant_text = "a" * 700

    asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(content=user_text),
                AIMessage(content=assistant_text),
                HumanMessage(content="follow up"),
            ]
        )
    )

    content = _prompt_content(model)
    assert f"User: {'u' * 300}" in content
    assert f"Assistant: {'a' * 500}" in content
    assert "u" * 301 not in content
    assert "a" * 501 not in content


def test_classifier_context_is_omitted_without_prior_messages():
    model = PromptCapturingModel(
        {"decision": "ALLOWED", "explanation": "Technical question."}
    )
    middleware = _middleware_with_model(model)

    asyncio.run(middleware._classify_query([HumanMessage(content="How do agents work?")]))

    assert _prompt_content(model) == (
        "Classify this user query for the LangChain documentation assistant. "
        "Consider both the text and any attached images. "
        "Return both the decision and one concise sentence explaining why."
        "\n\nUser query: How do agents work?"
    )


def test_anaphoric_follow_up_after_allowed_turn_is_allowed():
    model = PromptCapturingModel(
        {"decision": "ALLOWED", "explanation": "This refers to the prior answer."}
    )
    middleware = _middleware_with_model(model)

    result = asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(content="Explain StateGraph nodes and edges."),
                AIMessage(content="Nodes run work; edges connect their transitions."),
                HumanMessage(content="What do these two mean?"),
            ]
        )
    )

    assert result["decision"] == "ALLOWED"


def test_language_restatement_after_allowed_turn_is_allowed():
    model = PromptCapturingModel(
        {"decision": "ALLOWED", "explanation": "This restates the prior answer."}
    )
    middleware = _middleware_with_model(model)

    result = asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(content="Explain LangSmith tracing."),
                AIMessage(content="Tracing records runs, inputs, outputs, and metadata."),
                HumanMessage(content="Answer in Spanish."),
            ]
        )
    )

    assert result["decision"] == "ALLOWED"


def test_unrelated_weather_question_is_blocked():
    model = PromptCapturingModel(
        {"decision": "BLOCKED", "explanation": "Weather is outside scope."}
    )
    middleware = _middleware_with_model(model)

    result = asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(content="Explain LangSmith tracing."),
                AIMessage(content="Tracing records runs and metadata."),
                HumanMessage(content="Will it rain tomorrow?")
            ]
        )
    )

    assert result["decision"] == "BLOCKED"
