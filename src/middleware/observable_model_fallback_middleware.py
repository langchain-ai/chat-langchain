"""Model fallback middleware with provider failure observability."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import langsmith as ls
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langgraph.errors import GraphBubbleUp

logger = logging.getLogger(__name__)


def is_provider_auth_error(error: BaseException) -> bool:
    """Return whether an exception indicates invalid provider credentials."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in (401, 403):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "api_key_invalid",
            "api key not valid",
            "api key required",
            "unauthorized",
            "permission denied",
        )
    )


def _model_name(request: ModelRequest[Any]) -> str:
    return str(
        getattr(request.model, "model_name", None)
        or getattr(request.model, "model", None)
        or type(request.model).__name__
    )


def _record_auth_failure(request: ModelRequest[Any], error: Exception, logger: Any) -> None:
    model_name = _model_name(request)
    logger.error("Primary model %s failed authentication: %s", model_name, error)
    try:
        run_tree = ls.get_current_run_tree()
        if run_tree:
            run_tree.metadata["served_by_fallback"] = True
            run_tree.metadata["primary_model_error"] = str(error)
    except Exception:
        logger.debug("Unable to record provider fallback metadata", exc_info=True)


class ObservableModelFallbackMiddleware(ModelFallbackMiddleware):
    """Record authentication failures before using the normal fallback chain."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Observe authentication failures before running fallback models."""
        primary_attempt = True

        def recording_handler(current_request: ModelRequest[Any]) -> ModelResponse[Any]:
            nonlocal primary_attempt
            is_primary_attempt = primary_attempt
            primary_attempt = False
            try:
                return handler(current_request)
            except GraphBubbleUp:
                raise
            except Exception as error:
                if is_primary_attempt and is_provider_auth_error(error):
                    _record_auth_failure(current_request, error, logger)
                raise

        return super().wrap_model_call(request, recording_handler)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Observe authentication failures before running fallback models asynchronously."""
        primary_attempt = True

        async def recording_handler(
            current_request: ModelRequest[Any],
        ) -> ModelResponse[Any]:
            nonlocal primary_attempt
            is_primary_attempt = primary_attempt
            primary_attempt = False
            try:
                return await handler(current_request)
            except GraphBubbleUp:
                raise
            except Exception as error:
                if is_primary_attempt and is_provider_auth_error(error):
                    _record_auth_failure(current_request, error, logger)
                raise

        return await super().awrap_model_call(request, recording_handler)
