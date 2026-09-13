"""Tests for provider response sanitization."""

import asyncio
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.response_sanitizer_middleware import ResponseSanitizerMiddleware


def _run_middleware(message):
    middleware = ResponseSanitizerMiddleware()
    request = SimpleNamespace(model=SimpleNamespace(model="gemini-test"))

    async def handler(_request):
        return ModelResponse(result=[message])

    return asyncio.run(middleware.awrap_model_call(request, handler)).result[0]


def test_removes_text_frame_prefix():
    result = _run_middleware(AIMessage(content="text\nThe answer."))

    assert result.content == "The answer."


def test_removes_end_frame_prefix_from_text_block():
    result = _run_middleware(
        AIMessage(content=[{"type": "text", "text": "end\nThe answer."}])
    )

    assert result.content == [{"type": "text", "text": "The answer."}]


def test_removes_trailing_base64_blob_with_digit_frame():
    blob = "A" * 64
    result = _run_middleware(AIMessage(content=f"The answer.\n7\n{blob}"))

    assert result.content == "The answer."


def test_leaves_clean_answer_unchanged():
    content = "The answer is clean."
    result = _run_middleware(AIMessage(content=content))

    assert result.content == content
