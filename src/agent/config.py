"""Shared configuration for all agents."""

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import dotenv
import langsmith as ls
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda

from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware
from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)
from src.middleware.tool_retry_middleware import ToolRetryMiddleware

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class ProviderKeyValidationError(RuntimeError):
    """Raised when the primary provider rejects authentication."""


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


def _is_authentication_failure(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status_code = status_code or getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return True

    message = str(error).lower()
    return (
        "api_key_invalid" in message
        or ("invalid_argument" in message and "api key" in message)
        or "unauthorized" in message
        or "permission denied" in message
    )


def validate_provider_keys() -> None:
    """Make an authenticated request for every configured model provider."""
    if os.getenv("SKIP_PROVIDER_KEY_CHECK") == "1":
        logger.info("Skipping provider key validation")
        return

    primary_auth_error: Exception | None = None
    for model in [DEFAULT_MODEL, *FALLBACK_MODELS]:
        try:
            init_chat_model(model=model.id).invoke("health check")
        except Exception as error:
            if _is_authentication_failure(error):
                logger.error(
                    "Authentication failed for provider key %s",
                    model.api_key_env,
                )
                if model is DEFAULT_MODEL:
                    primary_auth_error = error
            else:
                logger.warning(
                    "Provider key check failed for %s (%s)",
                    model.api_key_env,
                    type(error).__name__,
                )
    if primary_auth_error is not None:
        raise ProviderKeyValidationError(
            f"Authentication failed for {DEFAULT_MODEL.api_key_env}"
        ) from primary_auth_error


def _mark_fallback_served(primary_error: Exception) -> None:
    run_tree = ls.get_current_run_tree()
    if run_tree is None:
        return
    while run_tree.parent_run is not None:
        run_tree = run_tree.parent_run
    run_tree.metadata["served_by_fallback"] = True
    run_tree.metadata["primary_model_failed"] = (
        "auth" if _is_authentication_failure(primary_error) else "error"
    )
    if "served_by_fallback" not in run_tree.tags:
        run_tree.tags.append("served_by_fallback")


class ObservableModelFallbackMiddleware(ModelFallbackMiddleware):
    """Mark the root trace when a fallback model serves the response."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Mark the root trace when a fallback model serves the response."""
        primary_error: Exception | None = None

        def tracked_handler(current_request: ModelRequest) -> ModelResponse:
            nonlocal primary_error
            if primary_error is not None:
                return handler(current_request)
            try:
                return handler(current_request)
            except Exception as error:
                primary_error = error
                raise

        response = super().wrap_model_call(request, tracked_handler)
        if primary_error is not None:
            _mark_fallback_served(primary_error)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Mark the root trace when an async fallback model serves the response."""
        primary_error: Exception | None = None

        async def tracked_handler(current_request: ModelRequest) -> ModelResponse:
            nonlocal primary_error
            if primary_error is not None:
                return await handler(current_request)
            try:
                return await handler(current_request)
            except Exception as error:
                primary_error = error
                raise

        response = await super().awrap_model_call(request, tracked_handler)
        if primary_error is not None:
            _mark_fallback_served(primary_error)
        return response


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

model_fallback_middleware = ObservableModelFallbackMiddleware(
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
    "ObservableModelFallbackMiddleware",
    "ProviderKeyValidationError",
    "validate_provider_keys",
    # Config
    "MAX_RETRIES",
    "logger",
]
