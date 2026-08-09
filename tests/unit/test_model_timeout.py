"""Tests that model calls are deadline-bounded and that timeouts reach the fallback chain."""

import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.agent import config
from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    is_retryable_exception,
)

TEST_TIMEOUT_S = 0.05

# Each provider stores the deadline under its own field name.
TIMEOUT_FIELDS = ("timeout", "request_timeout", "default_request_timeout")


def _configured_timeout(model):
    for field in TIMEOUT_FIELDS:
        value = getattr(model, field, None)
        if value is not None:
            return value
    return None


async def _stalled_model_call(_input):
    await asyncio.wait_for(asyncio.sleep(5), timeout=TEST_TIMEOUT_S)
    return AIMessage(content="never returned")


async def _fallback_model_call(_input):
    return AIMessage(content="fallback answer")


def test_models_are_deadline_bounded():
    """Every configured model must carry a per-request timeout."""
    assert config.MODEL_TIMEOUT_S > 0
    assert _configured_timeout(config.default_model) == config.MODEL_TIMEOUT_S

    fallback_models = config.model_fallback_middleware.models
    assert len(fallback_models) == len(config.FALLBACK_MODELS)
    assert [_configured_timeout(model) for model in fallback_models] == [
        config.MODEL_TIMEOUT_S
    ] * len(config.FALLBACK_MODELS)

    retrying_primary = config.summarization_model.runnable.bound.steps[0]
    assert _configured_timeout(retrying_primary) == config.MODEL_TIMEOUT_S


def test_provider_timeouts_are_classified_retryable():
    """Timeouts from either SDK family must be treated as retryable."""
    assert is_retryable_exception(asyncio.TimeoutError())
    assert is_retryable_exception(httpx.ReadTimeout("deadline exceeded"))
    assert not is_retryable_exception(ValueError("bad request"))


def test_stalled_model_raises_and_falls_back():
    """A model call that outlives its deadline should fail fast into the fallback."""
    primary = RunnableLambda(_stalled_model_call).with_retry(
        stop_after_attempt=config.MAX_RETRIES + 1
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(primary.ainvoke([HumanMessage(content="hi")]))

    chain = primary.with_fallbacks([RunnableLambda(_fallback_model_call)])
    result = asyncio.run(chain.ainvoke([HumanMessage(content="hi")]))

    assert result.content == "fallback answer"


def test_retry_middleware_retries_timeout_then_reraises():
    """The retry middleware retries a timeout, then re-raises so fallback can engage."""
    middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0.0)
    attempts = 0

    async def handler(request):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("deadline exceeded")

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(middleware.awrap_model_call(None, handler))

    assert attempts == 2
