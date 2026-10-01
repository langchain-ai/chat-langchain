import os
from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from src.agent import config


class FakeModel:
    def __init__(self, name):
        self.name = name


def test_authentication_fallback_marks_root_run(monkeypatch):
    root = SimpleNamespace(metadata={}, tags=[], parent_run=None)
    child = SimpleNamespace(metadata={}, tags=[], parent_run=root)
    monkeypatch.setattr(config.ls, "get_current_run_tree", lambda: child)

    middleware = config.ObservableModelFallbackMiddleware.__new__(
        config.ObservableModelFallbackMiddleware
    )
    middleware.models = [FakeModel("fallback")]
    request = ModelRequest(
        model=FakeModel("primary"),
        messages=[HumanMessage(content="hello")],
    )
    calls = 0

    def handler(current_request):
        nonlocal calls
        calls += 1
        if current_request.model.name == "primary":
            raise ValueError("HTTP 401 unauthorized")
        return AIMessage(content="fallback answer")

    response = middleware.wrap_model_call(request, handler)

    assert response.content == "fallback answer"
    assert calls == 2
    assert root.metadata == {
        "served_by_fallback": True,
        "primary_model_failed": "auth",
    }
    assert child.metadata == {}
    assert root.tags == ["served_by_fallback"]
