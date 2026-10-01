from types import SimpleNamespace

from langchain_core.messages import AIMessage

from src.middleware import primary_circuit_breaker_middleware as breaker_module
from src.middleware.primary_circuit_breaker_middleware import (
    PrimaryCircuitBreakerMiddleware,
)


class InvalidArgumentError(Exception):
    pass


def _request(model):
    return SimpleNamespace(
        model=model, override=lambda **kwargs: SimpleNamespace(model=kwargs["model"])
    )


def test_breaker_bypasses_primary_during_cooldown_and_marks_metadata(monkeypatch):
    primary = SimpleNamespace(model="gemini-3.5-flash-lite")
    middleware = PrimaryCircuitBreakerMiddleware(
        "google_genai:gemini-3.5-flash-lite",
        "openai:gpt-5.4-nano",
        cooldown_seconds=300,
    )
    metadata = {}
    monkeypatch.setattr("langgraph.config.get_config", lambda: {"metadata": metadata})
    calls = []

    def first_handler(request):
        calls.append(request.model)
        raise InvalidArgumentError("API_KEY_INVALID")

    try:
        middleware.wrap_model_call(_request(primary), first_handler)
    except InvalidArgumentError:
        pass

    result = middleware.wrap_model_call(
        _request(primary),
        lambda request: calls.append(request.model) or AIMessage(content="fallback"),
    )

    assert result.content == "fallback"
    assert [model.model for model in calls] == ["gemini-3.5-flash-lite", "gpt-5.4-nano"]
    assert metadata == {"served_by_fallback": True, "primary_failure_reason": "auth"}


def test_breaker_retries_primary_after_cooldown(monkeypatch):
    primary = SimpleNamespace(model="gemini-3.5-flash-lite")
    middleware = PrimaryCircuitBreakerMiddleware(
        "google_genai:gemini-3.5-flash-lite",
        "openai:gpt-5.4-nano",
        cooldown_seconds=5,
    )
    now = [100.0]
    monkeypatch.setattr(breaker_module.time, "monotonic", lambda: now[0])

    try:
        middleware.wrap_model_call(
            _request(primary),
            lambda _request: (_ for _ in ()).throw(
                InvalidArgumentError("API_KEY_INVALID")
            ),
        )
    except InvalidArgumentError:
        pass

    now[0] = 106.0
    calls = []
    middleware.wrap_model_call(
        _request(primary),
        lambda request: calls.append(request.model) or AIMessage(content="primary"),
    )

    assert [model.model for model in calls] == ["gemini-3.5-flash-lite"]
