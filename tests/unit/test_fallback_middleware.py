import logging

from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.middleware.fallback_middleware import ObservableModelFallbackMiddleware


def _request(model):
    return ModelRequest(model=model, messages=[])


def test_auth_failure_returns_fallback_and_records_metadata(monkeypatch, caplog):
    primary = FakeMessagesListChatModel(
        responses=[AIMessage(content="unused")], name="gemini-primary"
    )
    fallback = FakeMessagesListChatModel(
        responses=[AIMessage(content="fallback answer")], name="gpt-fallback"
    )
    middleware = ObservableModelFallbackMiddleware(fallback)
    run_tree = type("RunTree", (), {"metadata": {}})()
    monkeypatch.setattr(
        "src.middleware.fallback_middleware.ls.get_current_run_tree", lambda: run_tree
    )
    error = ValueError("API key not valid. Please pass a valid API key.")

    def handler(request):
        if request.model is primary:
            raise error
        return request.model.invoke([])

    with caplog.at_level(logging.ERROR):
        response = middleware.wrap_model_call(_request(primary), handler)

    assert response.content == "fallback answer"
    assert run_tree.metadata == {
        "served_by_model": "gpt-fallback",
        "primary_error_class": "ValueError",
    }
    assert "ValueError" in caplog.text
    assert "API key not valid" in caplog.text


def test_primary_success_has_no_fallback_metadata(monkeypatch):
    primary = FakeMessagesListChatModel(
        responses=[AIMessage(content="primary answer")], name="gemini-primary"
    )
    fallback = FakeMessagesListChatModel(
        responses=[AIMessage(content="unused")], name="gpt-fallback"
    )
    middleware = ObservableModelFallbackMiddleware(fallback)
    run_tree = type("RunTree", (), {"metadata": {}})()
    monkeypatch.setattr(
        "src.middleware.fallback_middleware.ls.get_current_run_tree", lambda: run_tree
    )

    response = middleware.wrap_model_call(
        _request(primary), lambda request: request.model.invoke([])
    )

    assert response.content == "primary answer"
    assert run_tree.metadata == {}
