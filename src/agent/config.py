"""Shared configuration for all agents."""

import logging
import os
import sys
from dataclasses import dataclass

import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda

from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware
from src.middleware.model_fallback import (
    AuthenticationAwareModelFallbackMiddleware,
    AuthenticationAwareRetry,
    AuthenticationAwareRunnableWithFallbacks,
    ModelAvailabilityState,
    is_authentication_failure,
)
from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
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

    id: str  # e.g., "google_genai:gemini-3.5-flash-lite"
    name: str  # Display name, e.g., "Gemini 3.5 Flash Lite"
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
    "gemini-3.5-flash-lite": ModelConfig(
        id="google_genai:gemini-3.5-flash-lite",
        name="Gemini 3.5 Flash Lite",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fastest, most cost-effective Gemini",
    ),
}

# Default models for different use cases
DEFAULT_MODEL = MODELS["gemini-3.5-flash-lite"]
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
MODEL_AUTH_COOLDOWN_SECONDS = float(os.getenv("MODEL_AUTH_COOLDOWN_SECONDS", "300"))
primary_model_availability = ModelAvailabilityState(MODEL_AUTH_COOLDOWN_SECONDS)

# Primary model. Public callers cannot switch this at runtime.
default_model = init_chat_model(model=DEFAULT_MODEL.id)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _run_model_preflight() -> None:
    """Check the configured primary model credentials with one minimal request."""
    if not os.getenv(DEFAULT_MODEL.api_key_env):
        return
    if "pytest" in sys.modules and os.getenv("MODEL_PREFLIGHT") != "1":
        return
    try:
        default_model.invoke("Reply with OK")
    except Exception as error:
        if is_authentication_failure(error):
            primary_model_availability.mark_unavailable()
            logger.error("Default model authentication preflight failed: %s", error)
            if os.getenv("STRICT_MODEL_PREFLIGHT") == "1":
                raise
        else:
            logger.warning(
                "Default model authentication preflight was inconclusive: %s", error
            )


_run_model_preflight()


def _raise_for_retryable_finish_reason(response: object) -> object:
    metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason", "")
    if finish_reason in RETRYABLE_FINISH_REASONS:
        raise MalformedResponseError(f"Model returned {finish_reason}")
    return response


def _init_retrying_model(model: str, track_auth: bool = False) -> Runnable:
    bound = init_chat_model(model=model) | RunnableLambda(
        _raise_for_retryable_finish_reason
    )
    if track_auth:
        return AuthenticationAwareRetry(
            bound=bound,
            max_attempt_number=MAX_RETRIES + 1,
            availability=primary_model_availability,
            model_id=model,
        )
    return _ProviderValidationAwareRunnableRetry(
        bound=bound,
        max_attempt_number=MAX_RETRIES + 1,
    )


def init_retry_fallback_model(model: str) -> Runnable:
    """Initialize a model runnable with the shared retry and fallback policy."""
    primary_model = _init_retrying_model(model, track_auth=model == DEFAULT_MODEL.id)
    fallback_models = [
        _init_retrying_model(fallback.id) for fallback in FALLBACK_MODELS
    ]
    if model == DEFAULT_MODEL.id:
        return AuthenticationAwareRunnableWithFallbacks(
            primary_model, fallback_models, primary_model_availability
        )
    return primary_model.with_fallbacks(fallback_models)


summarization_model = init_retry_fallback_model(DEFAULT_MODEL.id)

# =============================================================================
# Middleware
# =============================================================================

model_retry_middleware = ModelRetryMiddleware(max_retries=MAX_RETRIES)
tool_retry_middleware = ToolRetryMiddleware(max_attempts=3)
duplicate_call_guard_middleware = DuplicateCallGuardMiddleware()
docs_research_guard_middleware = DocsResearchGuardMiddleware()
citation_guard_middleware = CitationGuardMiddleware()
answer_sanity_guard_middleware = AnswerSanityGuardMiddleware()

model_fallback_middleware = AuthenticationAwareModelFallbackMiddleware(
    DEFAULT_MODEL.id,
    primary_model_availability,
    *[m.id for m in FALLBACK_MODELS],
)
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
    "duplicate_call_guard_middleware",
    "docs_research_guard_middleware",
    "citation_guard_middleware",
    "answer_sanity_guard_middleware",
    "model_fallback_middleware",
    # Config
    "MAX_RETRIES",
    "logger",
]
