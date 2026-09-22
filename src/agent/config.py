"""Shared configuration for all agents."""

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import (
    _sanitize_request_for_fallback,
    _supports_anthropic_cache_control,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.errors import GraphBubbleUp

from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware
from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
    _ProviderValidationAwareRunnableWithFallbacks,
    is_non_retryable_request_error,
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

# Primary model. Public callers cannot switch this at runtime.
default_model = init_chat_model(model=DEFAULT_MODEL.id)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _raise_for_retryable_finish_reason(response: object) -> object:
    metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason", "")
    if finish_reason in RETRYABLE_FINISH_REASONS:
        raise MalformedResponseError(f"Model returned {finish_reason}")
    return response


def _init_retrying_model(model: str) -> Runnable:
    return _ProviderValidationAwareRunnableRetry(
        bound=init_chat_model(model=model)
        | RunnableLambda(_raise_for_retryable_finish_reason),
        max_attempt_number=MAX_RETRIES + 1,
    )


def init_retry_fallback_model(model: str) -> Runnable:
    """Initialize a model runnable with the shared retry and fallback policy."""
    primary_model = _init_retrying_model(model)
    fallback_models = [
        _init_retrying_model(fallback.id) for fallback in FALLBACK_MODELS
    ]
    return _ProviderValidationAwareRunnableWithFallbacks(
        runnable=primary_model,
        fallbacks=fallback_models,
    )


class _ProviderValidationAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Stop model fallback on provider request-validation errors."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Run model fallbacks while preserving request-validation errors."""
        first_exception = None
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exception:
            if is_non_retryable_request_error(exception):
                raise
            first_exception = exception

        for fallback_model in self.models:
            fallback_request = (
                request
                if _supports_anthropic_cache_control(fallback_model)
                else _sanitize_request_for_fallback(request)
            )
            try:
                return handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as exception:
                if is_non_retryable_request_error(exception):
                    raise

        if first_exception is None:
            raise RuntimeError("No error stored at end of fallbacks.")
        raise first_exception

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Run async model fallbacks while preserving validation errors."""
        first_exception = None
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exception:
            if is_non_retryable_request_error(exception):
                raise
            first_exception = exception

        for fallback_model in self.models:
            fallback_request = (
                request
                if _supports_anthropic_cache_control(fallback_model)
                else _sanitize_request_for_fallback(request)
            )
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as exception:
                if is_non_retryable_request_error(exception):
                    raise

        if first_exception is None:
            raise RuntimeError("No error stored at end of fallbacks.")
        raise first_exception


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

model_fallback_middleware = _ProviderValidationAwareModelFallbackMiddleware(
    *[m.id for m in FALLBACK_MODELS]
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
