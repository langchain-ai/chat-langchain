import logging

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.fallback_visibility_middleware import FallbackVisibilityMiddleware


class _RunTree:
    def __init__(self):
        self.metadata = {}

    def add_metadata(self, metadata):
        self.metadata.update(metadata)


def test_authentication_fallback_is_logged_and_recorded(monkeypatch, caplog):
    primary = FakeListChatModel(responses=["unused"])
    fallback = FakeListChatModel(responses=["fallback"])
    run_tree = _RunTree()
    middleware = FallbackVisibilityMiddleware("primary-model", fallback)
    request = ModelRequest(
        model=primary,
        messages=[HumanMessage(content="Hi")],
    )

    def handler(current_request):
        if current_request.model is primary:
            raise RuntimeError("API_KEY_INVALID")
        return ModelResponse(result=[AIMessage(content="fallback")])

    monkeypatch.setattr(
        "src.middleware.fallback_visibility_middleware.get_current_run_tree",
        lambda: run_tree,
    )
    with caplog.at_level(logging.ERROR):
        response = middleware.wrap_model_call(request, handler)

    assert response.result[0].content == "fallback"
    assert "Primary model primary-model failed with RuntimeError" in caplog.text
    assert run_tree.metadata == {
        "served_by_fallback": True,
        "primary_error_type": "RuntimeError",
    }
