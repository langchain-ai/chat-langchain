"""Tests for summarization middleware retry/fallback wiring."""

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


def _assert_model_limits(model):
    tokens = getattr(model, "max_output_tokens", None) or getattr(
        model, "max_tokens", None
    )
    timeout = (
        getattr(model, "timeout", None)
        or getattr(model, "request_timeout", None)
        or getattr(model, "default_request_timeout", None)
    )
    assert tokens == config.MODEL_MAX_TOKENS
    assert timeout == config.MODEL_REQUEST_TIMEOUT


def test_retrying_models_have_effective_limits():
    models = [config.default_model, *config.model_fallback_middleware.models]
    for model_config in [config.DEFAULT_MODEL, *config.FALLBACK_MODELS]:
        models.append(config._init_retrying_model(model_config.id).bound.first)
    for model in models:
        _assert_model_limits(model)


def test_entrypoint_primary_has_effective_limits():
    import asyncio

    from langchain.agents.factory import _chain_async_model_call_handlers
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import AIMessage

    from agent import agent, docs_agent_middleware
    from src.middleware.model_fallback_middleware import (
        RepairingModelFallbackMiddleware,
    )
    from src.middleware.retry_middleware import ModelRetryMiddleware

    model_id = agent.config["model"]
    assert model_id == "google_genai:gemini-3.5-flash-lite"
    request = ModelRequest(model=init_chat_model(model_id), messages=[])
    chain = _chain_async_model_call_handlers(
        [
            m.awrap_model_call
            for m in docs_agent_middleware
            if isinstance(m, (RepairingModelFallbackMiddleware, ModelRetryMiddleware))
        ]
    )

    async def handler(request):
        _assert_model_limits(request.model)
        return ModelResponse(result=[AIMessage(content="Answer")])

    result = asyncio.run(chain(request, handler))
    assert result.model_response.result[-1].content == "Answer"
