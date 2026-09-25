"""Tests for forced documentation research model requests."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.retry_middleware import (
    DeterministicErrorAwareModelFallbackMiddleware,
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)


@pytest.mark.parametrize(
    "model_name",
    ["google_genai:gemini-3.5-flash-lite", "openai:gpt-5.4-nano"],
)
def test_forced_research_uses_provider_neutral_tool_choice(model_name):
    middleware = DocsResearchGuardMiddleware()
    calls = []
    request = ModelRequest(
        model=model_name,
        messages=[HumanMessage(content="How do I configure StateGraph?")],
    )

    async def handler(call):
        calls.append(call)
        if len(calls) == 1:
            return ModelResponse(
                result=[
                    AIMessage(content="StateGraph accepts a config_schema parameter.")
                ]
            )
        return ModelResponse(
            result=[
                AIMessage(
                    content="I need to verify that in the documentation.",
                    tool_calls=[
                        {
                            "name": "search_docs_by_lang_chain",
                            "args": {"query": "StateGraph"},
                            "id": "search-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert calls[1].tool_choice == "search_docs_by_lang_chain"
    assert not isinstance(calls[1].tool_choice, dict)
    assert response.result[0].tool_calls[0]["name"] == "search_docs_by_lang_chain"


@pytest.mark.parametrize(
    "error", [ValueError("invalid tool choice"), TypeError("bad request")]
)
def test_model_retry_does_not_retry_request_construction_errors(error):
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error), match=str(error)):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_fallback_does_not_mask_request_construction_error():
    middleware = DeterministicErrorAwareModelFallbackMiddleware(object())
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        raise ValueError("invalid tool choice")

    with pytest.raises(ValueError, match="invalid tool choice"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_runnable_retry_does_not_retry_request_construction_error():
    calls = 0

    def invoke(_request):
        nonlocal calls
        calls += 1
        raise ValueError("invalid tool choice")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(ValueError, match="invalid tool choice"):
        runnable.invoke("request")

    assert calls == 1
