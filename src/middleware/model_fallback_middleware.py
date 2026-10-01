"""Fallback middleware that exposes permanent model authentication failures."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import _sanitize_request_for_fallback
from langchain.agents.middleware.types import ModelRequest
from langgraph.errors import GraphBubbleUp
from langsmith.run_helpers import get_current_run_tree

_AUTH_FAILURE_REASONS = ("API_KEY_INVALID", "INVALID_API_KEY")


def _exception_text(error: BaseException) -> str:
    response = getattr(error, "response", None)
    response_text = getattr(response, "text", "") if response is not None else ""
    return f"{error} {response_text}".upper()


def auth_failure_reason(error: BaseException, model: object) -> str | None:
    """Return a stable reason for credential failures, if applicable."""
    text = _exception_text(error)
    status_code = getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )
    model_name = str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "model_id", None)
        or model
    ).lower()
    if any(reason in text for reason in _AUTH_FAILURE_REASONS) or (
        "INVALID_ARGUMENT" in text and "API" in text and "KEY" in text
    ):
        return "API_KEY_INVALID"
    if status_code in (401, 403) and any(
        provider in model_name for provider in ("openai", "anthropic")
    ):
        return f"HTTP_{status_code}"
    return None


def _model_name(model: object) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "model_id", None)
        or model
    )


def _record_fallback_metadata(reason: str, model: object) -> None:
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    run_tree.metadata.update(
        {
            "served_by_fallback": True,
            "fallback_reason": reason,
            "served_model": _model_name(model),
        }
    )


class ObservableModelFallbackMiddleware(ModelFallbackMiddleware):
    """Skip rejected primary credentials and record fallback responses."""

    _primary_auth_failure_reason: str | None = None
    _breaker_lock = threading.Lock()

    def __init__(self, first_model: str | Any, *additional_models: str | Any) -> None:
        """Initialize fallback models and their display names."""
        super().__init__(first_model, *additional_models)
        self._fallback_names = [_model_name(model) for model in self.models]

    def _primary_is_open(self) -> str | None:
        with self._breaker_lock:
            return type(self)._primary_auth_failure_reason

    def _trip_primary_breaker(self, reason: str) -> None:
        with self._breaker_lock:
            type(self)._primary_auth_failure_reason = reason

    @classmethod
    def reset_primary_breaker(cls) -> None:
        """Reset the process-level primary credential breaker."""
        with cls._breaker_lock:
            cls._primary_auth_failure_reason = None

    def _fallback_result(
        self, request: ModelRequest, handler: Callable, reason: str | None
    ):
        last_exception: Exception | None = None
        for fallback_model, fallback_name in zip(self.models, self._fallback_names):
            fallback_request = self._sanitize(request, fallback_model)
            try:
                result = handler(fallback_request.override(model=fallback_model))
                if reason is not None:
                    _record_fallback_metadata(reason, fallback_name)
                return result
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_exception = error
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    async def _async_fallback_result(
        self, request: ModelRequest, handler: Callable, reason: str | None
    ):
        last_exception: Exception | None = None
        for fallback_model, fallback_name in zip(self.models, self._fallback_names):
            fallback_request = self._sanitize(request, fallback_model)
            try:
                result = await handler(fallback_request.override(model=fallback_model))
                if reason is not None:
                    _record_fallback_metadata(reason, fallback_name)
                return result
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_exception = error
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    @staticmethod
    def _sanitize(request: ModelRequest, model: object) -> ModelRequest:
        return _sanitize_request_for_fallback(request, model)

    def wrap_model_call(self, request: ModelRequest, handler: Callable):
        """Run the primary model or an observable fallback."""
        reason = self._primary_is_open()
        if reason is not None:
            return self._fallback_result(request, handler, reason)
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            reason = auth_failure_reason(error, request.model)
            if reason is not None:
                self._trip_primary_breaker(reason)
            return self._fallback_result(request, handler, reason)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[..., Awaitable]):
        """Run the primary model or an observable async fallback."""
        reason = self._primary_is_open()
        if reason is not None:
            return await self._async_fallback_result(request, handler, reason)
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            reason = auth_failure_reason(error, request.model)
            if reason is not None:
                self._trip_primary_breaker(reason)
            return await self._async_fallback_result(request, handler, reason)
