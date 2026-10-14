"""Tests for summarization middleware retry/fallback wiring."""

import ast
import asyncio
import importlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.agent import config
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt


def test_summarization_middleware_uses_retrying_fallback_model():
    """Summarization should use the default model plus shared retry/fallback policy."""
    assert [model.id for model in config.FALLBACK_MODELS] == [
        "openai:gpt-5.4-nano",
        "anthropic:claude-haiku-4-5-20251001",
    ]

    middleware = CustomSummarizationMiddleware(
        model=config.DEFAULT_MODEL.id,
        summary_model=config.summarization_model,
        trigger=("tokens", 130_000),
        keep=("tokens", 30_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    )

    summary_model = middleware.summary_model
    summary_primary = getattr(summary_model, "runnable", None)
    summary_fallbacks = getattr(summary_model, "fallbacks", [])

    assert middleware.model.model == "gemini-3.5-flash-lite"
    assert type(summary_model).__name__ == "RunnableWithFallbacks"
    assert getattr(summary_primary, "max_attempt_number") == config.MAX_RETRIES + 1
    assert len(summary_fallbacks) == len(config.FALLBACK_MODELS)
    assert all(
        getattr(fallback, "max_attempt_number") == config.MAX_RETRIES + 1
        for fallback in summary_fallbacks
    )

    assert config.default_model.max_output_tokens == config.MAX_OUTPUT_TOKENS
    assert summary_primary.bound.first.max_output_tokens == config.MAX_OUTPUT_TOKENS
    assert all(
        fallback.bound.first.max_tokens == config.MAX_OUTPUT_TOKENS
        for fallback in summary_fallbacks
    )
    assert all(
        model.max_tokens == config.MAX_OUTPUT_TOKENS
        for model in config.model_fallback_middleware.models
    )
    assert config.model_retry_middleware.max_output_tokens == config.MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("configured_limit", [None, "1234"])
def test_output_limit_environment_configuration(monkeypatch, configured_limit):
    with monkeypatch.context() as environment:
        for key in config.API_KEYS:
            environment.setenv(key, "test")
        if configured_limit is None:
            environment.delenv("MODEL_MAX_OUTPUT_TOKENS", raising=False)
        else:
            environment.setenv("MODEL_MAX_OUTPUT_TOKENS", configured_limit)
        importlib.reload(config)
        expected = int(configured_limit or "8192")
        assert config.MAX_OUTPUT_TOKENS == expected
        assert config.default_model.max_output_tokens == expected
        assert config.model_retry_middleware.max_output_tokens == expected
        assert all(
            model.max_tokens == expected
            for model in config.model_fallback_middleware.models
        )
        assert (
            config.summarization_model.runnable.bound.first.max_output_tokens
            == expected
        )
        assert all(
            model.bound.first.max_tokens == expected
            for model in config.summarization_model.fallbacks
        )
    importlib.reload(config)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reason", ["MAX_TOKENS", "length", "repetition"])
def test_summary_generation_rejects_bad_output_and_uses_capped_fallbacks(
    monkeypatch, asynchronous, reason
):
    calls = []
    bad_message = AIMessage(
        content="*" * 5000 if reason == "repetition" else "partial answer",
        response_metadata={
            "finish_reason": "STOP" if reason == "repetition" else reason
        },
    )

    def init_model(*, model, max_tokens):
        def respond(prompt):
            calls.append((model, max_tokens))
            return (
                AIMessage(content="useful summary")
                if model == config.FALLBACK_MODELS[-1].id
                else bad_message
            )

        return RunnableLambda(respond)

    monkeypatch.setattr(config, "init_chat_model", init_model)
    monkeypatch.setattr(config, "MAX_OUTPUT_TOKENS", 1234)
    monkeypatch.setattr(config, "MAX_RETRIES", 0)
    middleware = CustomSummarizationMiddleware(
        model=config.default_model,
        summary_model=config.init_retry_fallback_model(config.DEFAULT_MODEL.id),
        trigger=("tokens", 130_000),
        keep=("tokens", 30_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    )
    messages = [HumanMessage(content="How do agents work?")]
    result = (
        asyncio.run(middleware._acreate_summary(messages))
        if asynchronous
        else middleware._create_summary(messages)
    )
    assert result == "useful summary"
    assert calls == [
        (model.id, 1234) for model in [config.DEFAULT_MODEL, *config.FALLBACK_MODELS]
    ]


def test_managed_agent_preserves_model_literal_and_fallback_wraps_retry(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_PROMPTS", "1")
    from agent import docs_agent_middleware

    middleware_types = [type(middleware) for middleware in docs_agent_middleware]
    assert middleware_types.index(
        type(config.model_fallback_middleware)
    ) < middleware_types.index(type(config.model_retry_middleware))

    module = ast.parse(Path("agent.py").read_text())
    definition = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "define_deep_agent"
    )
    model = next(
        keyword.value for keyword in definition.keywords if keyword.arg == "model"
    )
    assert isinstance(model, ast.Constant)
    assert model.value == config.DEFAULT_MODEL.id
