"""Record the fixed alias of the model that served the final response."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.utils.trace_root_metadata import update_root_run_metadata


class ServedModelFallbackMiddleware(ModelFallbackMiddleware):
    """Record which configured model produced the final response."""

    def _model_alias(self, request: ModelRequest, model: Any) -> str:
        if model is request.model:
            return "primary"
        for index, fallback_model in enumerate(self.models, start=1):
            if model is fallback_model:
                return f"fallback_{index}"
        return "primary"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Record the alias of the model that serves the response."""
        served_model = "primary"

        def tracked_handler(tracked_request: ModelRequest) -> ModelResponse:
            nonlocal served_model
            served_model = self._model_alias(request, tracked_request.model)
            return handler(tracked_request)

        response = super().wrap_model_call(request, tracked_handler)
        update_root_run_metadata(
            request.runtime,
            {"served_model": served_model},
        )
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Record the alias of the model that serves the response asynchronously."""
        served_model = "primary"

        async def tracked_handler(tracked_request: ModelRequest) -> ModelResponse:
            nonlocal served_model
            served_model = self._model_alias(request, tracked_request.model)
            return await handler(tracked_request)

        response = await super().awrap_model_call(request, tracked_handler)
        update_root_run_metadata(
            request.runtime,
            {"served_model": served_model},
        )
        return response


__all__ = ["ServedModelFallbackMiddleware"]
