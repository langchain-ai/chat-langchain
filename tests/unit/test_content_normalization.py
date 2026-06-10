"""Tests for cross-provider AIMessage content normalization."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.content_normalization_middleware import (
    ContentNormalizationMiddleware,
    normalize_message_content,
)


def test_normalizes_bare_string_inside_content_list():
    msg = AIMessage(content=[{"type": "text", "text": "hello"}, "bare string"])
    out = normalize_message_content([msg])
    assert out[0].content == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "bare string"},
    ]


def test_strips_google_extras_from_text_block():
    msg = AIMessage(
        content=[
            {
                "type": "text",
                "text": "from gemini",
                "extras": {"foo": "bar"},
                "index": 0,
            }
        ]
    )
    out = normalize_message_content([msg])
    assert out[0].content == [{"type": "text", "text": "from gemini"}]


def test_no_op_when_content_is_already_openai_compliant():
    msg = AIMessage(content=[{"type": "text", "text": "ok"}])
    out = normalize_message_content([msg])
    assert out[0].content == [{"type": "text", "text": "ok"}]
    assert out[0] is msg


def test_no_op_when_content_is_plain_string():
    msg = AIMessage(content="just a string")
    out = normalize_message_content([msg])
    assert out[0].content == "just a string"
    assert out[0] is msg


def test_passes_through_non_ai_messages():
    human = HumanMessage(content="hi")
    out = normalize_message_content([human])
    assert out[0] is human


def test_awrap_model_call_mutates_request_messages():
    class FakeRequest:
        def __init__(self, messages):
            self.messages = messages

    captured = {}

    async def handler(request):
        captured["messages"] = request.messages
        return "response"

    middleware = ContentNormalizationMiddleware()
    request = FakeRequest(
        [AIMessage(content=[{"type": "text", "text": "a"}, "bare"])]
    )

    result = asyncio.run(middleware.awrap_model_call(request, handler))
    assert result == "response"
    assert captured["messages"][0].content == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "bare"},
    ]
