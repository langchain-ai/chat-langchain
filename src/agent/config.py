# Shared configuration for all agents (models, middleware, API keys)
import logging
import os
from dataclasses import dataclass
from typing import Optional

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model

from src.middleware.retry_middleware import ModelRetryMiddleware

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class ModelConfig:
    id: str  # e.g., "google_genai:gemini-3.1-flash-lite-preview"
    name: str  # Display name, e.g., "Gemini 3.1 Flash Lite"
    provider: str  # e.g., "google", "openai", "baseten"
    api_key_env: str  # Environment variable for API key
    description: Optional[str] = None


# All backend-supported models. This intentionally mirrors the frontend's
# selectable model IDs plus Jewel's guardrails/fallback models.
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
    "gpt-5.4-mini": ModelConfig(
        id="openai:gpt-5.4-mini",
        name="GPT-5.4 Mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Strongest mini model for coding, computer use, and subagents",
    ),
    # Google
    "gemini-2.5-flash": ModelConfig(
        id="google_genai:gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fast and capable Google model",
    ),
    "gemini-3.1-flash-lite": ModelConfig(
        id="google_genai:gemini-3.1-flash-lite-preview",
        name="Gemini 3.1 Flash Lite",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fastest, most cost-effective Gemini",
    ),
    # Baseten
    "glm-5": ModelConfig(
        id="baseten:zai-org/GLM-5",
        name="GLM 5",
        provider="baseten",
        api_key_env="BASETEN_API_KEY",
        description="Z.ai GLM 5 served via Baseten",
    ),
}

# Default models for different use cases
DEFAULT_MODEL = MODELS["gemini-3.1-flash-lite"]
GUARDRAILS_MODEL = MODELS["gpt-5.4-nano"]

# Fallback chain (in order of preference)
FALLBACK_MODELS = [
    MODELS["gemini-2.5-flash"],
    MODELS["claude-haiku-4.5"],
]

# =============================================================================
# API Key Setup
# =============================================================================

API_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "BASETEN_API_KEY",
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

# Primary configurable model (can be switched at runtime)
configurable_model = init_chat_model(
    model=DEFAULT_MODEL.id,
    configurable_fields=("model",),
)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")

# =============================================================================
# Middleware
# =============================================================================

model_retry_middleware = ModelRetryMiddleware(max_retries=MAX_RETRIES)

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
    # Configurable models
    "configurable_model",
    # Middleware
    "model_retry_middleware",
    "model_fallback_middleware",
    # Config
    "MAX_RETRIES",
    "logger",
]
