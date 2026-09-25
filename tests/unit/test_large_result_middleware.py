import pytest
from langchain_core.messages import ToolMessage

from src.middleware.large_result_middleware import (
    LargeResultMiddleware,
    extract_large_result_path,
)


class Request:
    pass


@pytest.mark.parametrize(
    "content",
    [
        "Tool output was too large; see /large_tool_results/result-123.txt.",
        [
            {"type": "text", "text": "Output spilled."},
            {"type": "text", "text": "/large_tool_results/result-123.txt"},
        ],
    ],
)
def test_extract_large_result_path_from_string_or_structured_content(content):
    assert extract_large_result_path(content) == "/large_tool_results/result-123.txt"


@pytest.mark.anyio
async def test_middleware_appends_exact_path_read_instruction():
    message = ToolMessage(
        content="Output too large: /large_tool_results/result-123.txt",
        tool_call_id="call-1",
    )

    async def handler(_request):
        return message

    result = await LargeResultMiddleware().awrap_tool_call(Request(), handler)

    assert "/large_tool_results/result-123.txt" in result.content
    assert "Call read_file on exactly this path next" in result.content
    assert "offset=0" in result.content
    assert "limit=200" in result.content


@pytest.mark.anyio
async def test_middleware_passes_normal_tool_message_through_unchanged():
    message = ToolMessage(content="normal result", tool_call_id="call-1")

    async def handler(_request):
        return message

    result = await LargeResultMiddleware().awrap_tool_call(Request(), handler)

    assert result is message


@pytest.mark.anyio
async def test_middleware_recognizes_structured_spill_content():
    message = ToolMessage(
        content=[
            {"type": "text", "text": "Output too large."},
            {"type": "text", "text": "/large_tool_results/result-123.txt"},
        ],
        tool_call_id="call-1",
    )

    async def handler(_request):
        return message

    result = await LargeResultMiddleware().awrap_tool_call(Request(), handler)

    assert result.content[-1]["text"].startswith(
        "The tool output was too large and is stored at /large_tool_results/result-123.txt"
    )
