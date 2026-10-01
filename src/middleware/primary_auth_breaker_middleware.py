"""Primary model authentication breaker and fallback handling."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langgraph.errors import GraphBubbleUp
from langsmith import get_current_run_tree

logger = logging.getLogger(__name__)


class PrimaryAuthBreaker:
    """Track primary authentication failures for a process-local TTL."""

    def __init__(self, ttl: float = 300.0, clock: Callable[[], float] = time.monotonic):
        """Create a breaker with a configurable open interval."""
        self.ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._opened_until = 0.0

    def is_open(self) -> bool:
        """Return whether the breaker is currently open."""
        with self._lock:
            if self._opened_until <= self._clock():
                self._opened_until = 0.0
                return False
            return True

    def trip(self) -> bool:
        """Open the breaker and report whether this changed its state."""
        with self._lock:
            was_open = self._opened_until > self._clock()
            self._opened_until = self._clock() + self.ttl
            return not was_open


_AUTH_MARKERS = (
    "API_KEY_INVALID",
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "INVALID API KEY",
    "INVALID X-API-KEY",
    "AUTHENTICATION_ERROR",
    "AUTHENTICATION FAILED",
)


def is_primary_auth_error(error: BaseException) -> bool:
    """Return whether an exception indicates invalid primary credentials."""
    status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status_code in (401, 403):
        return True
    if status_code == 429 or isinstance(status_code, int) and status_code >= 500:
        return False
    text = str(error).upper()
    return any(marker in text for marker in _AUTH_MARKERS)


def _root_run_metadata() -> dict[str, Any] | None:
    run_tree = get_current_run_tree()
    if run_tree is None:
        return None
    while run_tree.parent_run is not None:
        run_tree = run_tree.parent_run
    return run_tree.metadata


class PrimaryAuthBreakerMiddleware(AgentMiddleware):
    """Skip an invalid primary and preserve the configured fallback order."""

    def __init__(
        self,
        primary_model: str,
        fallback_models: list[str],
        *,
        breaker: PrimaryAuthBreaker | None = None,
    ) -> None:
        """Configure the primary model and ordered fallbacks."""
        super().__init__()
        self.primary_model_id = primary_model
        self.fallback_model_ids = fallback_models
        self.fallback_models = [init_chat_model(model) for model in fallback_models]
        self.breaker = breaker or PrimaryAuthBreaker()

    def _set_served_model(self, model_id: str) -> None:
        metadata = _root_run_metadata()
        if metadata is not None:
            metadata["ls_served_model"] = model_id

    def _set_fallback_metadata(self, model_id: str, auth: bool) -> None:
        metadata = _root_run_metadata()
        if metadata is not None:
            metadata["served_by_fallback"] = True
            if auth:
                metadata["primary_error_kind"] = "auth"
            metadata["ls_served_model"] = model_id

    def _fallback_request(self, request: ModelRequest, model: Any) -> ModelRequest:
        return request.override(model=model)

    def _invoke_fallbacks(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        *,
        auth: bool,
    ) -> ModelResponse:
        last_exception: Exception | None = None
        for model, model_id in zip(self.fallback_models, self.fallback_model_ids):
            try:
                response = handler(self._fallback_request(request, model))
                self._set_fallback_metadata(model_id, auth)
                return response
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_exception = error
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    def wrap_model_call(self, request, handler):
        """Call the primary unless its auth breaker is active."""
        if self.breaker.is_open():
            return self._invoke_fallbacks(request, handler, auth=True)
        try:
            response = handler(request)
            self._set_served_model(self.primary_model_id)
            return response
        except GraphBubbleUp:
            raise
        except Exception as error:
            if not is_primary_auth_error(error):
                return self._invoke_fallbacks(request, handler, auth=False)
            if self.breaker.trip():
                logger.error(
                    "Primary model auth breaker opened for %s for %.0f seconds",
                    self.primary_model_id,
                    self.breaker.ttl,
                )
            return self._invoke_fallbacks(request, handler, auth=True)

    async def awrap_model_call(self, request, handler):
        """Call the primary asynchronously unless its auth breaker is active."""
        if self.breaker.is_open():
            return await self._ainvoke_fallbacks(request, handler, auth=True)
        try:
            response = await handler(request)
            self._set_served_model(self.primary_model_id)
            return response
        except GraphBubbleUp:
            raise
        except Exception as error:
            if not is_primary_auth_error(error):
                return await self._ainvoke_fallbacks(request, handler, auth=False)
            if self.breaker.trip():
                logger.error(
                    "Primary model auth breaker opened for %s for %.0f seconds",
                    self.primary_model_id,
                    self.breaker.ttl,
                )
            return await self._ainvoke_fallbacks(request, handler, auth=True)

    async def _ainvoke_fallbacks(self, request, handler, *, auth):
        last_exception: Exception | None = None
        for model, model_id in zip(self.fallback_models, self.fallback_model_ids):
            try:
                response = await handler(self._fallback_request(request, model))
                self._set_fallback_metadata(model_id, auth)
                return response
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_exception = error
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No fallback models configured")


def probe_primary_model(
    model: Any,
    model_id: str,
    api_key_env: str,
    breaker: PrimaryAuthBreaker,
) -> None:
    """Probe the primary credentials without blocking startup on failures."""
    if not os.getenv(api_key_env):
        logger.error("Primary model %s has no configured %s", model_id, api_key_env)
        return
    try:
        model.invoke("Reply with OK", config={"timeout": 1})
    except Exception as error:
        if is_primary_auth_error(error):
            logger.error("Primary model authentication failed: %s", model_id)
            if breaker.trip():
                logger.error(
                    "Primary model auth breaker opened for %s for %.0f seconds",
                    model_id,
                    breaker.ttl,
                )
        else:
            logger.warning("Primary model startup probe failed: %s", error)
