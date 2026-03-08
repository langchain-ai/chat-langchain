import pytest
from langsmith import testing as t


@pytest.mark.langsmith
def test_model_has_timeout_configured():
    """Model should have a request timeout configured to prevent indefinite hangs."""
    t.log_inputs({"test": "model_timeout_configuration"})

    from src.agent.config import configurable_model

    # Check that the model has some timeout configuration
    # The exact attribute depends on the implementation
    model_kwargs = getattr(configurable_model, "model_kwargs", {}) or {}
    timeout = (
        getattr(configurable_model, "request_timeout", None)
        or model_kwargs.get("timeout", None)
        or model_kwargs.get("request_timeout", None)
    )

    # For _ConfigurableModel, the timeout is stored in _default_config
    if timeout is None:
        default_config = getattr(configurable_model, "_default_config", {}) or {}
        timeout = default_config.get("timeout", None) or default_config.get(
            "request_timeout", None
        )

    t.log_outputs({"timeout_configured": timeout is not None, "timeout_value": str(timeout)})

    assert timeout is not None, (
        f"No timeout configured on model. "
        f"This causes grok-4-1-fast to hang for 1 hour before platform timeout kills it. "
        f"Model attrs: {[a for a in dir(configurable_model) if 'timeout' in a.lower()]}"
    )
    assert timeout <= 300, f"Timeout {timeout}s is too long (max 300s recommended)"
