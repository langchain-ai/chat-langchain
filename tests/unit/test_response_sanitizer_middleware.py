import asyncio
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.response_sanitizer_middleware import ResponseSanitizerMiddleware


def _run_middleware(message: AIMessage) -> AIMessage:
    middleware = ResponseSanitizerMiddleware()
    request = ModelRequest(
        model=SimpleNamespace(model_name="gemini-test"),
        messages=[],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[message])

    response = asyncio.run(middleware.awrap_model_call(request, handler))
    return response.result[0]


def test_strips_leading_text_part_label():
    message = AIMessage(
        content=[{"type": "text", "text": "text\nThe answer."}]
    )

    result = _run_middleware(message)

    assert result.content == [{"type": "text", "text": "The answer."}]


def test_strips_leading_end_part_label():
    message = AIMessage(content=[{"type": "text", "text": "end\nThe answer."}])

    result = _run_middleware(message)

    assert result.content == [{"type": "text", "text": "The answer."}]


def test_strips_trailing_signature_after_markdown_link_list():
    signature = "A" * 64 + "=="
    block = {
        "type": "text",
        "text": "- [Guide](https://example.com)\n7\n" + signature,
        "extras": {"signature": signature},
    }

    result = _run_middleware(AIMessage(content=[block]))

    assert result.content == [
        {
            "type": "text",
            "text": "- [Guide](https://example.com)",
            "extras": {"signature": signature},
        }
    ]


def test_clean_answer_passes_through_byte_identical():
    block = {"type": "text", "text": "A clean answer."}
    message = AIMessage(content=[block])

    result = _run_middleware(message)

    assert result.content == message.content
    assert result is message


def test_string_content_passes_through_unchanged():
    message = AIMessage(content="text\nA clean string answer.")

    result = _run_middleware(message)

    assert result.content == message.content
    assert result is message
