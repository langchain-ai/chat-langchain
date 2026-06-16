"""Shared configuration for all agents."""

import logging
import os
from dataclasses import dataclass

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda

from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
)
from src.middleware.tool_retry_middleware import ToolRetryMiddleware

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a supported chat model."""

    id: str  # e.g., "google_genai:gemini-3.1-flash-lite"
    name: str  # Display name, e.g., "Gemini 3.1 Flash Lite"
    provider: str  # e.g., "google", "openai", "baseten"
    api_key_env: str  # Environment variable for API key
    description: str | None = None


# Backend-supported models.
MODELS: dict[str, ModelConfig] = {
    # Anthropic
    "claude-haiku-4.5": ModelConfig(
        id="anthropic:claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="Fast and cheap Anthropic model",
    ),
    # OpenAI
    "gpt-5.4-nano": ModelConfig(
        id="openai:gpt-5.4-nano",
        name="GPT-5.4 Nano",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Cheapest GPT-5.4-class model for simple high-volume tasks",
    ),
    # Google
    "gemini-3.1-flash-lite": ModelConfig(
        id="google_genai:gemini-3.1-flash-lite",
        name="Gemini 3.1 Flash Lite",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fastest, most cost-effective Gemini",
    ),
}

# Default models for different use cases
DEFAULT_MODEL = MODELS["gemini-3.1-flash-lite"]
GUARDRAILS_MODEL = MODELS["gpt-5.4-nano"]

# Fallback chain (in order of preference)
FALLBACK_MODELS = [
    MODELS["gpt-5.4-nano"],
    MODELS["claude-haiku-4.5"],
]

# =============================================================================
# API Key Setup
# =============================================================================

API_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
]

for key in API_KEYS:
    if value := os.getenv(key):
        os.environ[key] = value.strip()
        logger.info(f"{key} configured")


# =============================================================================
# Model Initialization
# =============================================================================

# Retry configuration
MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "2"))

# Primary model. Public callers cannot switch this at runtime.
def _ls_metadata_for(model_id: str) -> dict[str, str]:
    """Return LangSmith-recognized provider/model metadata for a model id."""
    provider, _, name = model_id.partition(":")
    metadata = {"ls_model_type": "chat"}
    if provider:
        metadata["ls_provider"] = provider
    if name:
        metadata["ls_model_name"] = name
    return metadata


def _init_chat_model_with_ls_metadata(model_id: str):
    """Initialize a chat model and bind LangSmith provider/model metadata."""
    return init_chat_model(model=model_id).with_config(
        metadata=_ls_metadata_for(model_id)
    )


default_model = _init_chat_model_with_ls_metadata(DEFAULT_MODEL.id)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _raise_for_retryable_finish_reason(response: object) -> object:
    metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason", "")
    if finish_reason in RETRYABLE_FINISH_REASONS:
        raise MalformedResponseError(f"Model returned {finish_reason}")
    return response


def _init_retrying_model(model: str) -> Runnable:
    return (
        _init_chat_model_with_ls_metadata(model)
        | RunnableLambda(_raise_for_retryable_finish_reason)
    ).with_retry(stop_after_attempt=MAX_RETRIES + 1).with_config(
        metadata=_ls_metadata_for(model)
    )


def init_retry_fallback_model(model: str) -> Runnable:
    """Initialize a model runnable with the shared retry and fallback policy."""
    primary_model = _init_retrying_model(model)
    fallback_models = [_init_retrying_model(fallback.id) for fallback in FALLBACK_MODELS]
    return primary_model.with_fallbacks(fallback_models).with_config(
        metadata=_ls_metadata_for(model)
    )


summarization_model = init_retry_fallback_model(DEFAULT_MODEL.id)

# =============================================================================
# Middleware
# =============================================================================

model_retry_middleware = ModelRetryMiddleware(max_retries=MAX_RETRIES)
tool_retry_middleware = ToolRetryMiddleware(max_attempts=3)

model_fallback_middleware = ModelFallbackMiddleware(*[m.id for m in FALLBACK_MODELS])
logger.info(f"Fallback chain: {' -> '.join(m.name for m in FALLBACK_MODELS)}")

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "MODELS",
    "DEFAULT_MODEL",
    "GUARDRAILS_MODEL",
    "FALLBACK_MODELS",
    "ModelConfig",
    # Models
    "default_model",
    "init_retry_fallback_model",
    "summarization_model",
    # Middleware
    "model_retry_middleware",
    "tool_retry_middleware",
    "model_fallback_middleware",
    # Config
    "MAX_RETRIES",
    "logger",
]
