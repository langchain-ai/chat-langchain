from langchain.agents.middleware import ModelCallLimitMiddleware

from src.agent import config
from src.middleware.execution_limits_middleware import ExecutionTimeoutMiddleware


def test_server_limits_are_independent_of_caller_config():
    assert config.DEFAULT_RECURSION_LIMIT == 100
    assert isinstance(config.server_recursion_limit_middleware, ModelCallLimitMiddleware)
    assert config.server_recursion_limit_middleware.run_limit == 100
    assert config.server_recursion_limit_middleware.exit_behavior == "end"


def test_server_timeout_default_is_configurable(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_TIMEOUT_SECONDS", "42")
    middleware = ExecutionTimeoutMiddleware(42)

    assert middleware.timeout_seconds == 42
