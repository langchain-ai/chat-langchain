"""Tests for fixed served-model metadata aliases."""

import asyncio
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware import served_model_middleware as served_model_module
from src.middleware.served_model_middleware import ServedModelFallbackMiddleware


class FakeModel:
    _llm_type = "fake"


def test_fallback_records_alias_of_successful_model(monkeypatch):
    primary = FakeModel()
    fallback_one = FakeModel()
    fallback_two = FakeModel()
    middleware = ServedModelFallbackMiddleware.__new__(ServedModelFallbackMiddleware)
    middleware.models = [fallback_one, fallback_two]
    updates = []
    monkeypatch.setattr(
        served_model_module,
        "update_root_run_metadata",
        lambda runtime, metadata: updates.append(metadata),
    )
    request = ModelRequest(
        model=primary,
        messages=[HumanMessage(content="Question")],
        runtime=SimpleNamespace(),
    )

    async def handler(tracked_request):
        if tracked_request.model is not fallback_two:
            raise RuntimeError("model unavailable")
        return ModelResponse(result=[AIMessage(content="answer")])

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert updates == [{"served_model": "fallback_2"}]
