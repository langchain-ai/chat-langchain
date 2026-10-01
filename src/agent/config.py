"""Shared configuration for all agents."""

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda
from langsmith.run_helpers import set_run_metadata

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
    value = os.getenv(key, "").strip()
    if value:
        os.environ[key] = value
        logger.info(f"{key} configured")


def _log_primary_model_key_status() -> None:
    if not os.getenv(DEFAULT_MODEL.api_key_env, "").strip():
        logger.error(
            "%s is not configured for the primary model", DEFAULT_MODEL.api_key_env
        )


_log_primary_model_key_status()


# =============================================================================
# Model Initialization
# =============================================================================

# Retry configuration
MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "2"))

# Primary model. Public callers cannot switch this at runtime.
default_model = init_chat_model(model=DEFAULT_MODEL.id)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _validate_primary_model_key() -> None:
    if os.getenv("VALIDATE_MODEL_KEYS") != "1":
        return

    try:
        default_model.invoke("ping")
    except Exception as exc:
        logger.error(
            "Primary model %s rejected %s: %s",
            DEFAULT_MODEL.name,
            DEFAULT_MODEL.api_key_env,
            exc,
        )
        if os.getenv("VALIDATE_MODEL_KEYS_FAIL_FAST") == "1":
            raise


_validate_primary_model_key()


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


def _is_authentication_failure(exception: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).upper()
        if "API_KEY_INVALID" in message:
            return True
        for candidate in (current, getattr(current, "response", None)):
            status = getattr(candidate, "status_code", None)
            if status in (401, 403):
                return True
        current = current.__cause__ or current.__context__
    return False


class PrimaryModelFallbackMiddleware(ModelFallbackMiddleware):
    """Record primary authentication failures before using model fallbacks."""

    _auth_failure_logged = False
    _auth_failure_lock = Lock()

    @classmethod
    def _record_auth_failure(cls) -> None:
        with cls._auth_failure_lock:
            if not cls._auth_failure_logged:
                logger.error(
                    "Primary model authentication failed for %s (%s); using fallback models",
                    DEFAULT_MODEL.name,
                    DEFAULT_MODEL.provider,
                )
                cls._auth_failure_logged = True
        set_run_metadata(primary_model_error="auth", served_by_fallback=True)

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> ModelCallResult:
        primary_attempt = True

        def recording_handler(current_request: ModelRequest) -> ModelCallResult:
            nonlocal primary_attempt
            try:
                return handler(current_request)
            except Exception as exc:
                if primary_attempt and _is_authentication_failure(exc):
                    self._record_auth_failure()
                primary_attempt = False
                raise

        return super().wrap_model_call(request, recording_handler)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Any
    ) -> ModelCallResult:
        primary_attempt = True

        async def recording_handler(current_request: ModelRequest) -> ModelCallResult:
            nonlocal primary_attempt
            try:
                return await handler(current_request)
            except Exception as exc:
                if primary_attempt and _is_authentication_failure(exc):
                    self._record_auth_failure()
                primary_attempt = False
                raise

        return await super().awrap_model_call(request, recording_handler)


model_fallback_middleware = PrimaryModelFallbackMiddleware(
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
