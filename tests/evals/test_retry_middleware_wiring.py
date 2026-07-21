# tests/evals/test_retry_middleware_wiring.py
import pytest
from langsmith import testing as t

from agent import docs_agent_middleware
from src.agent import config
from src.middleware.retry_middleware import ModelRetryMiddleware


@pytest.mark.langsmith
def test_retry_middleware_is_in_agent_middleware_list():
    """Ensure ModelRetryMiddleware is wired into the MDA middleware stack."""

    t.log_inputs({"check": "ModelRetryMiddleware in docs_agent_middleware"})

    types = [type(m).__name__ for m in docs_agent_middleware]
    t.log_outputs({"middleware_types": types})
    t.log_reference_outputs({"expected": "ModelRetryMiddleware in middleware"})

    assert any(isinstance(m, ModelRetryMiddleware) for m in docs_agent_middleware), (
        f"ModelRetryMiddleware not found in docs_agent_middleware. "
        f"Found: {types}"
    )


@pytest.mark.langsmith
def test_retry_middleware_config_is_exported():
    """Ensure model_retry_middleware is properly exported from config."""

    t.log_inputs({"check": "model_retry_middleware exported from config"})

    assert hasattr(config, "model_retry_middleware"), "model_retry_middleware not in config"
    assert isinstance(config.model_retry_middleware, ModelRetryMiddleware)
    t.log_outputs({"result": "model_retry_middleware found and is correct type"})
    t.log_reference_outputs({"expected": "model_retry_middleware exported and correct type"})
