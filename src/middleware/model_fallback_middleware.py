"""Credential-aware model fallback middleware."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import _sanitize_request_for_fallback
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp

logger = logging.getLogger(__name__)

DEFAULT_AUTH_COOLDOWN_SECONDS = 600.0


def _provider_for_model(model: object) -> str:
    provider = str(
        getattr(model, "_llm_type", None)
        or getattr(model, "model_provider", None)
        or ""
    )
    if provider:
        if provider.startswith("google"):
            return "google_genai"
        if provider.startswith("openai"):
            return "openai"
        if provider.startswith("anthropic"):
            return "anthropic"
    model_id = getattr(model, "model", None) or getattr(model, "model_name", None)
    return str(model_id or "").split(":", 1)[0]


def is_permanent_credential_error(error: BaseException, provider: str) -> bool:
    """Return whether an error indicates rejected provider credentials."""
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    text = str(error).lower()
    if provider == "google_genai":
        return (
            "api_key_invalid" in text
            or (status_code == 403 and "permission_denied" in text)
            or (status_code == 400 and "api_key_invalid" in text)
        )
    if provider == "openai":
        return status_code == 401 and "invalid_api_key" in text
    if provider == "anthropic":
        return status_code == 401 and "authentication_error" in text
    return False


class CredentialAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Skip a credential-failed primary model during a cooldown window."""

    def __init__(
        self,
        first_model: str,
        *additional_models: str,
        cooldown_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure fallback models and credential cooldown tracking."""
        super().__init__(first_model, *additional_models)
        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else float(
                os.getenv("MODEL_AUTH_COOLDOWN_SECONDS", str(DEFAULT_AUTH_COOLDOWN_SECONDS))
            )
        )
        self._clock = clock
        self._unhealthy_until: dict[str, float] = {}

    def _model_id(self, model: object) -> str:
        return str(
            getattr(model, "model", None)
            or getattr(model, "model_name", None)
            or model
        )

    def _primary_is_unhealthy(self, model_id: str) -> bool:
        expires_at = self._unhealthy_until.get(model_id, 0.0)
        if expires_at <= self._clock():
            self._unhealthy_until.pop(model_id, None)
            return False
        return True

    def _record_primary_failure(self, model: object, error: BaseException) -> None:
        model_id = self._model_id(model)
        provider = _provider_for_model(model)
        if not is_permanent_credential_error(error, provider):
            return
        if not self._primary_is_unhealthy(model_id):
            self._unhealthy_until[model_id] = self._clock() + self.cooldown_seconds
            logger.error(
                "Primary model %s rejected credentials; using fallback for %.0f seconds",
                model_id,
                self.cooldown_seconds,
            )

    def _prepare_fallback_metadata(
        self, model: object
    ) -> tuple[dict[str, Any] | None, bool]:
        try:
            config = get_config()
        except RuntimeError:
            return None, False
        metadata = config.setdefault("metadata", {})
        metadata.setdefault("primary_model_failed", True)
        if "fallback_served" in metadata:
            return metadata, False
        metadata["fallback_served"] = self._model_id(model)
        return metadata, True

    def _clear_fallback_metadata(
        self, metadata: dict[str, Any] | None, added: bool
    ) -> None:
        if added and metadata is not None:
            metadata.pop("fallback_served", None)
            metadata.pop("primary_model_failed", None)

    def _fallback_sync(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        last_exception: BaseException | None = None,
    ) -> ModelResponse:
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(
                request, fallback_model
            ).override(model=fallback_model)
            metadata, metadata_added = self._prepare_fallback_metadata(fallback_model)
            try:
                return handler(fallback_request)
            except GraphBubbleUp:
                self._clear_fallback_metadata(metadata, metadata_added)
                raise
            except Exception as error:
                self._clear_fallback_metadata(metadata, metadata_added)
                last_exception = error
        if last_exception is None:
            raise RuntimeError("No fallback models configured")
        raise last_exception

    async def _fallback_async(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        last_exception: BaseException | None = None,
    ) -> ModelResponse:
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(
                request, fallback_model
            ).override(model=fallback_model)
            metadata, metadata_added = self._prepare_fallback_metadata(fallback_model)
            try:
                return await handler(fallback_request)
            except GraphBubbleUp:
                self._clear_fallback_metadata(metadata, metadata_added)
                raise
            except Exception as error:
                self._clear_fallback_metadata(metadata, metadata_added)
                last_exception = error
        if last_exception is None:
            raise RuntimeError("No fallback models configured")
        raise last_exception

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Run the primary model or the first available fallback."""
        model_id = self._model_id(request.model)
        if self._primary_is_unhealthy(model_id):
            return self._fallback_sync(request, handler)
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            self._record_primary_failure(request.model, error)
            return self._fallback_sync(request, handler, error)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Run the primary model or the first available async fallback."""
        model_id = self._model_id(request.model)
        if self._primary_is_unhealthy(model_id):
            return await self._fallback_async(request, handler)
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            self._record_primary_failure(request.model, error)
            return await self._fallback_async(request, handler, error)


__all__ = [
    "CredentialAwareModelFallbackMiddleware",
    "is_permanent_credential_error",
]
