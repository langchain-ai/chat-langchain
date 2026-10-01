"""Observable model fallback middleware."""

import logging

import langsmith as ls
from langchain.agents.middleware import ModelFallbackMiddleware

logger = logging.getLogger(__name__)


def _model_name(model: object) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "name", None)
        or model.__class__.__name__
    )


def _is_authentication_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status_code in {401, 403}:
        return True

    error_text = f"{error.__class__.__name__} {error}".lower()
    return any(
        marker in error_text
        for marker in (
            "api_key_invalid",
            "api key not valid",
            "authenticationerror",
            "authentication error",
            "unauthenticated",
            "permissiondenied",
            "permission denied",
            "forbidden",
        )
    )


def _record_fallback_metadata(model: object, error: Exception) -> None:
    try:
        run_tree = ls.get_current_run_tree()
        if run_tree:
            run_tree.metadata["served_by_model"] = _model_name(model)
            run_tree.metadata["primary_error_class"] = error.__class__.__name__
    except Exception:
        logger.debug("Unable to record model fallback metadata", exc_info=True)


class ObservableModelFallbackMiddleware(ModelFallbackMiddleware):
    """Log primary failures and annotate successful fallback responses."""

    def _log_primary_failure(self, error: Exception) -> None:
        log = logger.error if _is_authentication_error(error) else logger.warning
        log("Primary model failed with %s: %s", error.__class__.__name__, error)

    def wrap_model_call(self, request, handler):
        """Log primary failures and record the model serving fallback answers."""
        primary_error: Exception | None = None
        served_model: object = request.model

        def observed_handler(observed_request):
            nonlocal primary_error, served_model
            try:
                response = handler(observed_request)
                served_model = observed_request.model
                return response
            except Exception as error:
                if primary_error is None:
                    primary_error = error
                    self._log_primary_failure(error)
                raise

        response = super().wrap_model_call(request, observed_handler)
        if primary_error is not None:
            _record_fallback_metadata(served_model, primary_error)
        return response

    async def awrap_model_call(self, request, handler):
        """Log primary failures and record fallback answers asynchronously."""
        primary_error: Exception | None = None
        served_model: object = request.model

        async def observed_handler(observed_request):
            nonlocal primary_error, served_model
            try:
                response = await handler(observed_request)
                served_model = observed_request.model
                return response
            except Exception as error:
                if primary_error is None:
                    primary_error = error
                    self._log_primary_failure(error)
                raise

        response = await super().awrap_model_call(request, observed_handler)
        if primary_error is not None:
            _record_fallback_metadata(served_model, primary_error)
        return response


__all__ = ["ObservableModelFallbackMiddleware"]
