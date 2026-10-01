"""Shared configuration for all agents."""

import logging
import os
from dataclasses import dataclass

import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware
from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    AuthenticationAwareModelFallbackMiddleware,
    AuthenticationAwareRunnable,
    MalformedResponseError,
    ModelRetryMiddleware,
    PrimaryModelAuthCircuit,
    _ProviderValidationAwareRunnableRetry,
    is_authentication_error,
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

PRIMARY_MODEL_AUTH_COOLDOWN_SECONDS = float(
    os.getenv("PRIMARY_MODEL_AUTH_COOLDOWN_SECONDS", "300")
)
primary_model_auth_circuit = PrimaryModelAuthCircuit(
    DEFAULT_MODEL.provider,
    DEFAULT_MODEL.id,
    PRIMARY_MODEL_AUTH_COOLDOWN_SECONDS,
)

# Primary model. Public callers cannot switch this at runtime.
_primary_api_key = os.getenv(DEFAULT_MODEL.api_key_env, "").strip()
default_model = (
    init_chat_model(model=DEFAULT_MODEL.id) if _primary_api_key else None
)
if default_model is not None:
    logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _startup_auth_validation_skipped() -> bool:
    return os.getenv("PRIMARY_MODEL_AUTH_VALIDATION_SKIP", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _validate_primary_model_auth() -> bool:
    api_key = os.getenv(DEFAULT_MODEL.api_key_env, "").strip()
    if not api_key:
        logger.error(
            "Primary model authentication is unavailable: %s is not set for %s:%s",
            DEFAULT_MODEL.api_key_env,
            DEFAULT_MODEL.provider,
            DEFAULT_MODEL.id,
        )
        primary_model_auth_circuit.trip(RuntimeError("missing API key"))
        return False
    if _startup_auth_validation_skipped():
        logger.warning(
            "Skipping startup authentication validation for %s:%s",
            DEFAULT_MODEL.provider,
            DEFAULT_MODEL.id,
        )
        return True
    try:
        if default_model is None:
            return False
        default_model.invoke([HumanMessage(content="health check")])
    except Exception as exception:
        if is_authentication_error(exception):
            logger.error(
                "Primary model authentication validation failed for %s:%s: %s",
                DEFAULT_MODEL.provider,
                DEFAULT_MODEL.id,
                exception,
            )
            primary_model_auth_circuit.trip(exception)
            return False
        logger.warning(
            "Primary model authentication validation could not complete for %s:%s: %s",
            DEFAULT_MODEL.provider,
            DEFAULT_MODEL.id,
            exception,
        )
    return True


PRIMARY_MODEL_AUTH_OK = _validate_primary_model_auth()


def _raise_for_retryable_finish_reason(response: object) -> object:
    metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason", "")
    if finish_reason in RETRYABLE_FINISH_REASONS:
        raise MalformedResponseError(f"Model returned {finish_reason}")
    return response


def _init_retrying_model(model: str) -> Runnable:
    bound = (
        init_chat_model(model=model)
        if model != DEFAULT_MODEL.id or default_model is not None
        else RunnableLambda(
            lambda _: (_ for _ in ()).throw(
                RuntimeError(f"{DEFAULT_MODEL.api_key_env} is not configured")
            )
        )
    )
    retrying_model = _ProviderValidationAwareRunnableRetry(
        bound=bound
        | RunnableLambda(_raise_for_retryable_finish_reason),
        max_attempt_number=MAX_RETRIES + 1,
    )
    if model == DEFAULT_MODEL.id:
        return AuthenticationAwareRunnable(retrying_model, primary_model_auth_circuit)
    return retrying_model


def init_retry_fallback_model(model: str) -> Runnable:
    """Initialize a model runnable with the shared retry and fallback policy."""
    primary_model = _init_retrying_model(model)
    fallback_models = [
        _init_retrying_model(fallback.id) for fallback in FALLBACK_MODELS
    ]
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
    primary_model_auth_circuit,
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
    "PRIMARY_MODEL_AUTH_OK",
    "PRIMARY_MODEL_AUTH_COOLDOWN_SECONDS",
    "logger",
]
