"""Tests for ModelRetryMiddleware content normalization.

P0 Bug: ChatGoogleGenerativeAI with Gemini 3.x models returns
AIMessage.content as a list of content-part dicts (e.g.
[{"text": "...", "extras": {"signature": "..."}, "index": 0}]).
This list format propagated as the final message content, causing users to
receive malformed output.

Root cause: src/middleware/retry_middleware.py — nothing normalized list
content to a string before returning the response.

Fix: ModelRetryMiddleware._normalize_content() joins plain text blocks into a
single string while leaving structured outputs (thinking, tool_use) untouched.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


# ---------------------------------------------------------------------------
# Minimal stubs — avoid importing LangChain types directly in tests
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ModelResponse / AIMessage."""

    def __init__(self, content, response_metadata=None):
        self.content = content
        self.response_metadata = response_metadata or {}

    def model_copy(self, *, update=None):
        data = {
            "content": self.content,
            "response_metadata": self.response_metadata,
        }
        if update:
            data.update(update)
        return _FakeResponse(
            content=data["content"],
            response_metadata=data["response_metadata"],
        )


class _FakeRequest:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_middleware() -> ModelRetryMiddleware:
    return ModelRetryMiddleware(max_retries=0)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# _normalize_content unit tests
# ---------------------------------------------------------------------------


class TestNormalizeContent:
    def test_plain_string_content_unchanged(self):
        mw = _make_middleware()
        resp = _FakeResponse(content="hello world")
        result = mw._normalize_content(resp)
        assert result.content == "hello world"

    def test_gemini_list_content_normalized_to_string(self):
        """Reproduces the production Gemini 3.x list-content bug."""
        mw = _make_middleware()
        resp = _FakeResponse(
            content=[{"text": "hello world", "extras": {"signature": "abc123"}, "index": 0}]
        )
        result = mw._normalize_content(resp)
        assert result.content == "hello world"

    def test_multiple_text_blocks_joined(self):
        mw = _make_middleware()
        resp = _FakeResponse(
            content=[{"text": "foo"}, {"text": "bar"}]
        )
        result = mw._normalize_content(resp)
        assert result.content == "foobar"

    def test_thinking_block_not_normalized(self):
        """Lists containing 'thinking' blocks must be left as-is."""
        mw = _make_middleware()
        original_content = [
            {"type": "thinking", "thinking": "let me reason..."},
            {"type": "text", "text": "the answer"},
        ]
        resp = _FakeResponse(content=original_content)
        result = mw._normalize_content(resp)
        assert result.content is original_content

    def test_tool_use_block_not_normalized(self):
        """Lists containing 'tool_use' blocks must be left as-is."""
        mw = _make_middleware()
        original_content = [
            {"type": "tool_use", "name": "search", "input": {"query": "langchain"}},
        ]
        resp = _FakeResponse(content=original_content)
        result = mw._normalize_content(resp)
        assert result.content is original_content

    def test_empty_list_unchanged(self):
        mw = _make_middleware()
        resp = _FakeResponse(content=[])
        result = mw._normalize_content(resp)
        assert result.content == []

    def test_block_without_text_key_unchanged(self):
        mw = _make_middleware()
        original_content = [{"some_other_key": "value"}]
        resp = _FakeResponse(content=original_content)
        result = mw._normalize_content(resp)
        assert result.content is original_content


# ---------------------------------------------------------------------------
# awrap_model_call integration test
# ---------------------------------------------------------------------------


class TestAwrapModelCall:
    @pytest.mark.asyncio
    async def test_list_content_normalized_in_awrap(self):
        """awrap_model_call must normalize Gemini list content to a string."""
        gemini_response = _FakeResponse(
            content=[{"text": "hello world", "extras": {"signature": "abc123"}, "index": 0}]
        )
        handler = AsyncMock(return_value=gemini_response)

        mw = _make_middleware()
        result = await mw.awrap_model_call(_FakeRequest(), handler)
        assert result.content == "hello world"

    @pytest.mark.asyncio
    async def test_thinking_content_not_normalized_in_awrap(self):
        """awrap_model_call must NOT normalize responses with thinking blocks."""
        original_content = [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "answer"},
        ]
        response = _FakeResponse(content=original_content)
        handler = AsyncMock(return_value=response)

        mw = _make_middleware()
        result = await mw.awrap_model_call(_FakeRequest(), handler)
        assert result.content is original_content
