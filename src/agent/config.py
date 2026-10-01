"""Shared configuration for all agents."""

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import _sanitize_request_for_fallback
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.errors import GraphBubbleUp
from langsmith.run_helpers import get_current_run_tree

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
_credential_error_logged = False
_credential_error_log_lock = Lock()

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


def _is_credential_error(exc: BaseException) -> bool:
    """Return whether an exception indicates invalid or unauthorized credentials."""
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if candidate in (401, 403, "401", "403"):
            return True

    error_text = str(exc)
    return any(
        marker in error_text
        for marker in (
            "API_KEY_INVALID",
            "PERMISSION_DENIED",
            "API key not valid",
            "invalid x-api-key",
        )
    )


def _model_id(model: object) -> str:
    """Return the configured model identifier for a chat model."""
    return str(
        getattr(model, "model", None)
        or getattr(model, "model_name", None)
        or getattr(model, "model_id", None)
        or model
    )


def _stamp_credential_failure(primary_model_id: str) -> None:
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.add_metadata(
            {"primary_model_failed": "auth", "primary_model": primary_model_id}
        )


def _log_credential_failure(primary_model_id: str, api_key_env: str) -> None:
    global _credential_error_logged
    with _credential_error_log_lock:
        if not _credential_error_logged:
            logger.error(
                "Primary model %s failed credential validation; using fallback (%s)",
                primary_model_id,
                api_key_env,
            )
            _credential_error_logged = True


class CredentialAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Fallback middleware that records primary credential failures."""

    def __init__(self, first_model: object, *additional_models: object) -> None:
        """Initialize fallback model identifiers and instances."""
        self.fallback_model_ids = [
            model if isinstance(model, str) else None
            for model in (first_model, *additional_models)
        ]
        super().__init__(first_model, *additional_models)

    def _handle_primary_error(self, exc: Exception) -> bool:
        primary_model_id = DEFAULT_MODEL.id
        if _is_credential_error(exc):
            _log_credential_failure(primary_model_id, DEFAULT_MODEL.api_key_env)
            _stamp_credential_failure(primary_model_id)
            return True
        return False

    def _run_fallbacks(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        last_exception: Exception,
        credential_failure: bool,
    ) -> ModelResponse | Any:
        for index, fallback_model in enumerate(self.models):
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                response = handler(fallback_request.override(model=fallback_model))
                run_tree = get_current_run_tree()
                if credential_failure and run_tree is not None:
                    served_model = self.fallback_model_ids[index] or _model_id(
                        fallback_model
                    )
                    run_tree.add_metadata({"served_model": served_model})
                return response
            except GraphBubbleUp:
                raise
            except Exception as exc:
                last_exception = exc
        raise last_exception

    async def _arun_fallbacks(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        last_exception: Exception,
        credential_failure: bool,
    ) -> ModelResponse | Any:
        for index, fallback_model in enumerate(self.models):
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                response = await handler(fallback_request.override(model=fallback_model))
                run_tree = get_current_run_tree()
                if credential_failure and run_tree is not None:
                    served_model = self.fallback_model_ids[index] or _model_id(
                        fallback_model
                    )
                    run_tree.add_metadata({"served_model": served_model})
                return response
            except GraphBubbleUp:
                raise
            except Exception as exc:
                last_exception = exc
        raise last_exception

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> Any:
        """Try the primary model once, then fall back after errors."""
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            credential_failure = self._handle_primary_error(exc)
            return self._run_fallbacks(request, handler, exc, credential_failure)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable
    ) -> Any:
        """Try the primary model once, then fall back after errors asynchronously."""
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            credential_failure = self._handle_primary_error(exc)
            return await self._arun_fallbacks(
                request, handler, exc, credential_failure
            )


def preflight_model_keys() -> None:
    """Make one minimal request to each configured model."""
    for model_config in MODELS.values():
        if not model_config.api_key_env:
            continue
        try:
            init_chat_model(model=model_config.id).invoke("ping")
        except Exception as exc:
            if _is_credential_error(exc):
                raise RuntimeError(
                    f"Credential preflight failed for {model_config.id} "
                    f"using {model_config.api_key_env}"
                ) from exc
            raise


if os.getenv("MODEL_KEY_PREFLIGHT") == "1":
    preflight_model_keys()


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

model_fallback_middleware = CredentialAwareModelFallbackMiddleware(
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
    "CredentialAwareModelFallbackMiddleware",
    "_is_credential_error",
    "preflight_model_keys",
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
