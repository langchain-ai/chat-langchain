"""Tests for model timeout wiring."""

from agent import agent
from src.agent import config


def _model_timeout(model):
    """Return the provider-specific timeout field from a chat model."""
    return next(
        value
        for name in ("timeout", "request_timeout", "default_request_timeout")
        if (value := getattr(model, name, None)) is not None
    )


def test_primary_and_fallback_models_have_timeout():
    """Every model in the agent model paths has the shared timeout."""
    assert _model_timeout(config.default_model) == config.MODEL_TIMEOUT_SECONDS
    assert agent.config["model"] is config.default_model

    retrying_model = config._init_retrying_model(config.DEFAULT_MODEL.id)
    assert _model_timeout(retrying_model.bound.first) == config.MODEL_TIMEOUT_SECONDS
    assert all(
        _model_timeout(model) == config.MODEL_TIMEOUT_SECONDS
        for model in config.model_fallback_middleware.models
    )


def test_retry_fallback_models_have_timeout():
    """Retry and fallback runnables retain the shared timeout."""
    retry_fallback = config.init_retry_fallback_model(config.DEFAULT_MODEL.id)

    models = [retry_fallback.runnable, *retry_fallback.fallbacks]
    assert all(
        _model_timeout(model.bound.first) == config.MODEL_TIMEOUT_SECONDS
        for model in models
    )
