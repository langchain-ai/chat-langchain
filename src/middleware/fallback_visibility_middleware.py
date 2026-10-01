"""Middleware for making authentication fallbacks visible."""

import logging
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langsmith.run_helpers import get_current_run_tree

from src.middleware.model_error_utils import is_auth_or_permission_error

logger = logging.getLogger(__name__)


class FallbackVisibilityMiddleware(ModelFallbackMiddleware):
    """Record authentication errors when a fallback successfully serves a call."""

    def __init__(
        self,
        primary_model_id: str,
        *fallback_models: str | BaseChatModel,
    ) -> None:
        """Configure the primary model identifier and fallback models."""
        super().__init__(*fallback_models)
        self.primary_model_id = primary_model_id

    def _record_fallback(self, error: Exception) -> None:
        logger.error(
            "Primary model %s failed with %s; serving the step with a fallback",
            self.primary_model_id,
            type(error).__name__,
        )
        try:
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.add_metadata(
                    {
                        "served_by_fallback": True,
                        "primary_error_type": type(error).__name__,
                    }
                )
        except Exception:
            logger.debug("Unable to attach fallback metadata", exc_info=True)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Record authentication failures before successful fallback calls."""
        primary_error: Exception | None = None

        def tracking_handler(tracked_request: ModelRequest) -> ModelResponse:
            nonlocal primary_error
            try:
                return handler(tracked_request)
            except Exception as error:
                if tracked_request is request and is_auth_or_permission_error(error):
                    primary_error = error
                raise

        response = super().wrap_model_call(request, tracking_handler)
        if primary_error is not None:
            self._record_fallback(primary_error)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Record authentication failures before successful async fallback calls."""
        primary_error: Exception | None = None

        async def tracking_handler(tracked_request: ModelRequest) -> ModelResponse:
            nonlocal primary_error
            try:
                return await handler(tracked_request)
            except Exception as error:
                if tracked_request is request and is_auth_or_permission_error(error):
                    primary_error = error
                raise

        response = await super().awrap_model_call(request, tracking_handler)
        if primary_error is not None:
            self._record_fallback(primary_error)
        return response
