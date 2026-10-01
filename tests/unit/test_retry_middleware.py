import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    ObservableModelFallbackMiddleware,
    _ProviderValidationAwareRunnableRetry,
)


class _FakeModel(BaseChatModel):
    name: str

    @property
    def _llm_type(self):
        return self.name

    @property
    def _identifying_params(self):
        return {"name": self.name}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.name))]
        )


class _FakeRunTree:
    def __init__(self):
        self.metadata = {}
        self.tags = []
        self.patched = False

    def add_metadata(self, metadata):
        self.metadata.update(metadata)

    def add_tags(self, tags):
        self.tags.extend(tags)

    def patch(self):
        self.patched = True


def test_provider_validation_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    with pytest.raises(ValueError, match="unsupported request shape"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_provider_auth_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise RuntimeError("API_KEY_INVALID")

    with pytest.raises(RuntimeError, match="API_KEY_INVALID"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_model_retry_wrapper_does_not_retry_provider_validation_error():
    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(ValueError, match="unsupported request shape"):
        runnable.invoke("request")

    assert calls == 1


def test_auth_failure_serves_fallback_and_records_metadata(monkeypatch, caplog):
    primary = _FakeModel(name="google_genai:gemini-3.5-flash-lite")
    fallback = _FakeModel(name="openai:gpt-5.4-nano")
    middleware = ObservableModelFallbackMiddleware.__new__(
        ObservableModelFallbackMiddleware
    )
    middleware.models = [fallback]
    run_tree = _FakeRunTree()
    monkeypatch.setattr(
        "src.middleware.retry_middleware.get_current_run_tree", lambda: run_tree
    )

    def handler(request):
        if request.model is primary:
            raise RuntimeError("400 INVALID_ARGUMENT API_KEY_INVALID")
        return AIMessage(content="fallback answer")

    with caplog.at_level("ERROR"):
        response = middleware.wrap_model_call(
            ModelRequest(model=primary, messages=[HumanMessage(content="Hi")]),
            handler,
        )

    assert response.content == "fallback answer"
    assert run_tree.metadata == {
        "served_by_fallback": True,
        "primary_error": "auth",
    }
    assert run_tree.patched
    assert (
        "Primary model google_genai:gemini-3.5-flash-lite failed (auth)" in caplog.text
    )


def test_success_does_not_record_fallback_metadata(monkeypatch):
    primary = _FakeModel(name="google_genai:gemini-3.5-flash-lite")
    fallback = _FakeModel(name="openai:gpt-5.4-nano")
    middleware = ObservableModelFallbackMiddleware.__new__(
        ObservableModelFallbackMiddleware
    )
    middleware.models = [fallback]
    run_tree = _FakeRunTree()
    monkeypatch.setattr(
        "src.middleware.retry_middleware.get_current_run_tree", lambda: run_tree
    )

    response = middleware.wrap_model_call(
        ModelRequest(model=primary, messages=[HumanMessage(content="Hi")]),
        lambda request: AIMessage(content="primary answer"),
    )

    assert response.content == "primary answer"
    assert run_tree.metadata == {}
    assert not run_tree.patched
