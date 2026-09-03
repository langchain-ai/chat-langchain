"""Tests for the retrieval gate middleware."""

from __future__ import annotations

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.middleware.retrieval_gate_middleware import RetrievalGateMiddleware


def _make_request(messages: list) -> ModelRequest:
    return ModelRequest(
        model=None,
        messages=messages,
        system_message=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": messages},
        runtime=None,
        model_settings={},
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_reinvokes_when_substantive_answer_without_retrieval():
    middleware = RetrievalGateMiddleware()
    calls: list[ModelRequest] = []

    fabricated = AIMessage(
        content="Here is how:\n```python\nfrom langchain import x\n```\n"
        "See docs.langchain.com/foo for more."
    )
    grounded = AIMessage(content="Short grounded answer.")

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(result=[fabricated])
        return ModelResponse(result=[grounded])

    request = _make_request([HumanMessage(content="How do I use middleware?")])
    result = _run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert result.result[-1] is grounded
    injected = calls[1].messages[-1]
    assert isinstance(injected, HumanMessage)
    assert "did not call any docs retrieval tool" in injected.content


def test_passes_through_when_retrieval_happened_this_turn():
    middleware = RetrievalGateMiddleware()
    calls: list[ModelRequest] = []

    answer = AIMessage(
        content="```python\ncode\n```\n" + "x" * 400
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[answer])

    messages = [
        HumanMessage(content="How do I use middleware?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_docs_by_lang_chain", "args": {}, "id": "1"}
            ],
        ),
    ]
    result = _run(middleware.awrap_model_call(_make_request(messages), handler))

    assert len(calls) == 1
    assert result.result[-1] is answer


def test_passes_through_short_non_technical_answer():
    middleware = RetrievalGateMiddleware()
    calls: list[ModelRequest] = []

    greeting = AIMessage(content="Hi! How can I help with LangChain today?")

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[greeting])

    request = _make_request([HumanMessage(content="hello")])
    result = _run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert result.result[-1] is greeting


def test_second_attempt_returned_as_is():
    middleware = RetrievalGateMiddleware()
    calls: list[ModelRequest] = []

    fabricated = AIMessage(content="docs.langchain.com/again " + "y" * 400)

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[fabricated])

    request = _make_request([HumanMessage(content="explain streaming")])
    result = _run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert result.result[-1] is fabricated
