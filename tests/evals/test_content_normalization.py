# Tests for ContentNormalizerMiddleware
import pytest
from langchain_core.messages import AIMessage

from src.middleware.content_normalizer_middleware import (
    ContentNormalizerMiddleware,
    _extract_text,
)

# ---------------------------------------------------------------------------
# Unit tests for _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_single_block_with_signature():
    """Single Anthropic content block with extras.signature → plain string."""
    content = [
        {
            "type": "text",
            "text": "hello",
            "extras": {"signature": "EjQKMgG+Pvb7xtgk..."},
            "index": 0,
        }
    ]
    assert _extract_text(content) == "hello"


def test_extract_text_multiple_text_blocks():
    """Multiple text blocks are concatenated."""
    content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    assert _extract_text(content) == "hello world"


def test_extract_text_plain_string_passthrough():
    """Plain string content is returned unchanged."""
    assert _extract_text("already a string") == "already a string"


def test_extract_text_ignores_non_text_blocks():
    """Non-text blocks (e.g. tool_use) are skipped."""
    content = [
        {"type": "tool_use", "id": "toolu_abc", "name": "Search", "input": {}},
        {"type": "text", "text": "result text"},
    ]
    assert _extract_text(content) == "result text"


def test_extract_text_empty_list():
    """Empty list yields empty string."""
    assert _extract_text([]) == ""


def test_extract_text_list_with_bare_strings():
    """List that contains bare string elements (not dicts)."""
    content = ["hello ", "world"]
    assert _extract_text(content) == "hello world"


def test_extract_text_non_string_non_list_fallback():
    """Fallback for unexpected types — converts to str."""
    assert _extract_text(42) == "42"


# ---------------------------------------------------------------------------
# Middleware integration tests (no live model needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_normalizes_list_content():
    """aafter_model normalises AIMessage list content to a plain string."""
    middleware = ContentNormalizerMiddleware()
    state = {
        "messages": [
            AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "**actual response**",
                        "extras": {"signature": "EjQKMgG+Pvb7xtgk..."},
                        "index": 0,
                    }
                ]
            )
        ]
    }
    result = await middleware.aafter_model(state, runtime=None)
    assert result is not None
    assert result["messages"][0].content == "**actual response**"


@pytest.mark.asyncio
async def test_middleware_leaves_string_content_alone():
    """aafter_model does not modify already-string AIMessage content."""
    middleware = ContentNormalizerMiddleware()
    state = {
        "messages": [
            AIMessage(content="plain string response")
        ]
    }
    result = await middleware.aafter_model(state, runtime=None)
    # Either None (no change) or same string
    if result is not None:
        assert result["messages"][0].content == "plain string response"


@pytest.mark.asyncio
async def test_middleware_no_messages():
    """aafter_model returns None when state has no messages."""
    middleware = ContentNormalizerMiddleware()
    result = await middleware.aafter_model({"messages": []}, runtime=None)
    assert result is None


# ---------------------------------------------------------------------------
# LangSmith integration test (requires LANGSMITH_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.langsmith
def test_content_normalization_langsmith():
    """LangSmith-tagged smoke test for the normalization function."""
    # Simulates the broken payload observed in 23% of traces
    broken_content = [
        {
            "extras": {"signature": "EjQKMgG+Pvb7xtgk..."},
            "index": 0,
            "text": "**actual response**",
            "type": "text",
        }
    ]
    result = _extract_text(broken_content)
    assert result == "**actual response**", (
        f"Expected plain string, got: {result!r}"
    )
