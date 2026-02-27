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
    id: str  # e.g., "xai:grok-4-1-fast-non-reasoning"
    name: str  # Display name, e.g., "Grok 4.1 Fast"
    provider: str  # e.g., "xai", "anthropic", "openai"
    api_key_env: str  # Environment variable for API key
    description: Optional[str] = None


# All available models - single source of truth
MODELS: dict[str, ModelConfig] = {
    # xAI (Grok)
    "grok-4.1-fast": ModelConfig(
        id="xai:grok-4-1-fast-non-reasoning",
        name="Grok 4.1 Fast",
        provider="xai",
        api_key_env="XAI_API_KEY",
        description="Fast, optimized for tool calling",
    ),
    # Anthropic
    "claude-haiku": ModelConfig(
        id="anthropic:claude-haiku-4-5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="Fast and cost-effective",
    ),
    "claude-sonnet": ModelConfig(
        id="anthropic:claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="Balanced performance",
    ),
    "claude-opus": ModelConfig(
        id="anthropic:claude-opus-4-5",
        name="Claude Opus 4.5",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="Most capable Anthropic model",
    ),
    # OpenAI
    "gpt-5-nano": ModelConfig(
        id="openai:gpt-5-nano",
        name="GPT-5 Nano",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Fastest, most economical",
    ),
    "gpt-5-mini": ModelConfig(
        id="openai:gpt-5-mini",
        name="GPT-5 Mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Fast and capable",
    ),
    "gpt-5": ModelConfig(
        id="openai:gpt-5",
        name="GPT-5",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Most capable OpenAI model",
    ),
    # Google
    "gemini-2.5-flash": ModelConfig(
        id="google_genai:gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fast and capable Google model",
    ),
    "gemini-3-flash": ModelConfig(
        id="google_genai:gemini-3-flash-preview",
        name="Gemini 3 Flash",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Latest Gemini 3 Flash model",
    ),
    "gemini-3-pro": ModelConfig(
        id="google_genai:gemini-3-pro-preview",
        name="Gemini 3 Pro",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Most capable Gemini model",
    ),
}

# Preferred order for default and guardrails (first available wins)
_DEFAULT_MODEL_ORDER = ["claude-haiku", "grok-4.1-fast", "gpt-5-mini", "claude-sonnet", "gemini-2.5-flash"]
_GUARDRAILS_MODEL_ORDER = ["grok-4.1-fast", "claude-haiku", "gpt-5-mini", "claude-sonnet", "gemini-2.5-flash"]
_FALLBACK_CHAIN_ORDER = [
    MODELS["claude-haiku"],
    MODELS["grok-4.1-fast"],
    MODELS["gpt-5-mini"],
    MODELS["claude-sonnet"],
]

# =============================================================================
# API Key Setup
# =============================================================================

API_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
    "GOOGLE_API_KEY",
]

for key in API_KEYS:
    if value := os.getenv(key):
        os.environ[key] = value.strip()
        logger.info(f"{key} configured")


def _has_api_key(model_config: ModelConfig) -> bool:
    """True if the model's API key env var is set (non-empty)."""
    return bool(os.getenv(model_config.api_key_env, "").strip())


# Only use models whose API keys are set
def _first_available(order: list[str]) -> Optional[ModelConfig]:
    for key in order:
        m = MODELS[key]
        if _has_api_key(m):
            return m
    return None


def _filter_available(model_list: list[ModelConfig]) -> list[ModelConfig]:
    return [m for m in model_list if _has_api_key(m)]


# Resolve default and guardrails from configured providers only
_resolved_default = _first_available(_DEFAULT_MODEL_ORDER)
if _resolved_default is None:
    raise RuntimeError(
        "No LLM API key is set. Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, GOOGLE_API_KEY. "
        "See .env.example for details."
    )
DEFAULT_MODEL = _resolved_default

_resolved_guardrails = _first_available(_GUARDRAILS_MODEL_ORDER)
GUARDRAILS_MODEL = _resolved_guardrails if _resolved_guardrails is not None else DEFAULT_MODEL
if _resolved_guardrails is None:
    logger.info(f"No separate guardrails key; using default model for guardrails: {GUARDRAILS_MODEL.name}")

FALLBACK_MODELS = _filter_available(_FALLBACK_CHAIN_ORDER)
if not FALLBACK_MODELS:
    FALLBACK_MODELS = [DEFAULT_MODEL]
    logger.info("Fallback chain has only one model (no other API keys configured)")

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

# Anthropic-optimized model with caching (only if Anthropic key is set)
anthropic_configurable_model = None
if _has_api_key(MODELS["claude-sonnet"]):
    anthropic_configurable_model = init_chat_model(
        model=MODELS["claude-sonnet"].id,
        configurable_fields=("model",),
    )

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
    "anthropic_configurable_model",
    # Middleware
    "model_retry_middleware",
    "model_fallback_middleware",
    # Config
    "MAX_RETRIES",
    "logger",
]
