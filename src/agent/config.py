"""Shared configuration for all agents."""

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Awaitable, Callable

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langsmith import run_helpers

from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware
from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
    is_provider_auth_error,
)
from src.middleware.tool_retry_middleware import ToolRetryMiddleware

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
_auth_failure_logged = False
_auth_failure_log_lock = Lock()


def _record_primary_auth_fallback(request: ModelRequest) -> None:
    global _auth_failure_logged
    model_id = getattr(request.model, "model", None) or getattr(
        request.model, "model_name", type(request.model).__name__
    )
    provider = next(
        (
            model.provider
            for model in MODELS.values()
            if model.id.split(":", 1)[-1] == model_id
        ),
        getattr(request.model, "_llm_type", type(request.model).__name__),
    )
    with _auth_failure_log_lock:
        if not _auth_failure_logged:
            logger.error(
                "Primary model authentication failed (provider=%s, model=%s); "
                "serving fallback. Correct the provider credentials.",
                provider,
                model_id,
            )
            _auth_failure_logged = True
    run_tree = run_helpers.get_current_run_tree()
    if run_tree is not None:
        while run_tree.parent_run is not None:
            run_tree = run_tree.parent_run
        run_tree.add_metadata({"primary_model_auth_failed": True})


class _AuthAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Record auth-triggered synchronous fallback without changing the chain."""
        primary_auth_failed = False

        def tracked_handler(model_request: ModelRequest) -> ModelResponse:
            nonlocal primary_auth_failed
            try:
                return handler(model_request)
            except Exception as exc:
                if model_request.model is request.model and is_provider_auth_error(exc):
                    primary_auth_failed = True
                raise

        result = super().wrap_model_call(request, tracked_handler)
        if primary_auth_failed:
            _record_primary_auth_fallback(request)
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Record auth-triggered asynchronous fallback without changing the chain."""
        primary_auth_failed = False

        async def tracked_handler(model_request: ModelRequest) -> ModelResponse:
            nonlocal primary_auth_failed
            try:
                return await handler(model_request)
            except Exception as exc:
                if model_request.model is request.model and is_provider_auth_error(exc):
                    primary_auth_failed = True
                raise

        result = await super().awrap_model_call(request, tracked_handler)
        if primary_auth_failed:
            _record_primary_auth_fallback(request)
        return result


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

model_fallback_middleware = _AuthAwareModelFallbackMiddleware(
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
