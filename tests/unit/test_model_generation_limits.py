"""Test bounded model configuration and shared summary validation."""

import ast
import asyncio
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent import agent, docs_agent_middleware
from src.agent import config
from src.middleware.retry_middleware import (
    MalformedResponseError,
    ModelRecoveryMiddleware,
    ModelRetryMiddleware,
)


def _assert_limits(model):
    assert (
        getattr(model, "max_output_tokens", getattr(model, "max_tokens", None))
        == config.MAX_OUTPUT_TOKENS
    )
    timeout = next(
        getattr(model, name)
        for name in ("timeout", "request_timeout", "default_request_timeout")
        if getattr(model, name, None) is not None
    )
    assert timeout == config.REQUEST_TIMEOUT
    assert model.max_retries == 0
    assert model.disable_streaming is True


def test_primary_and_actual_answer_fallbacks_are_bounded():
    _assert_limits(config.default_model)
    assert config.model_recovery_middleware.primary_model is config.default_model
    assert len(config.model_fallback_middleware.models) == len(config.FALLBACK_MODELS)
    for model in config.model_fallback_middleware.models:
        _assert_limits(model)


def test_summary_primary_and_fallbacks_are_bounded():
    summary = config.summarization_model
    for runnable in [summary.runnable, *summary.fallbacks]:
        _assert_limits(runnable.bound.first)
        assert runnable.max_attempt_number == config.MAX_RETRIES + 1


def test_deployment_model_literal_and_runtime_middleware_order():
    model_id = "google_genai:gemini-3.5-flash-lite"
    assert agent.config["model"] == model_id
    tree = ast.parse(Path("agent.py").read_text())
    definition = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "define_deep_agent"
    )
    model_argument = next(
        keyword.value for keyword in definition.keywords if keyword.arg == "model"
    )
    assert isinstance(model_argument, ast.Constant)
    assert model_argument.value == model_id
    recovery = next(
        index
        for index, middleware in enumerate(docs_agent_middleware)
        if isinstance(middleware, ModelRecoveryMiddleware)
    )
    retry = next(
        index
        for index, middleware in enumerate(docs_agent_middleware)
        if isinstance(middleware, ModelRetryMiddleware)
    )
    fallback = docs_agent_middleware.index(config.model_fallback_middleware)
    assert recovery < fallback < retry


def test_configurable_limits_apply_to_initialized_models(monkeypatch):
    monkeypatch.setattr(config, "MAX_OUTPUT_TOKENS", 2048)
    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 15)
    for model in [config.DEFAULT_MODEL, *config.FALLBACK_MODELS]:
        _assert_limits(config._init_bounded_model(model.id))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
@pytest.mark.parametrize(
    "metadata", [{"finish_reason": "STOP"}, {"stop_reason": "max_tokens"}]
)
def test_summary_validation_retries_then_falls_back(
    monkeypatch, asynchronous, exhausted, metadata
):
    bad = AIMessage(content="Title\n" + "*   " * 5000, response_metadata=metadata)
    good = AIMessage(content="A concise context summary.")
    primary = FakeMessagesListChatModel(responses=[bad])
    fallback = FakeMessagesListChatModel(responses=[bad if exhausted else good])
    models = iter([primary, fallback, fallback])
    monkeypatch.setattr(config, "_init_bounded_model", lambda model: next(models))
    monkeypatch.setattr(config, "MAX_RETRIES", 1)
    summary = config.init_retry_fallback_model(config.DEFAULT_MODEL.id)
    for runnable in [summary.runnable, *summary.fallbacks]:
        runnable.wait_exponential_jitter = False

    def invoke():
        return (
            asyncio.run(summary.ainvoke("Summarize"))
            if asynchronous
            else summary.invoke("Summarize")
        )

    if exhausted:
        with pytest.raises(MalformedResponseError):
            invoke()
    else:
        assert invoke().content == good.content
