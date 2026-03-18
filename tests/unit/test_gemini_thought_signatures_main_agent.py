"""Tests for Gemini thought-signature normalization in ModelRetryMiddleware.

These tests verify that when a Gemini model returns AI message content as a
list of dicts (with embedded thought-signature metadata), the
ModelRetryMiddleware strips that structure and returns a plain string to callers.

No network access or LangSmith credentials are required — all model calls are
mocked.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _normalize_content,
    _normalize_response,
)

try:
    from langchain.agents.middleware.types import ModelResponse
except ImportError:
    ModelResponse = None


# ---------------------------------------------------------------------------
# Unit tests for _normalize_content
# ---------------------------------------------------------------------------


class TestNormalizeContent(unittest.TestCase):
    def test_plain_string_unchanged(self):
        result = _normalize_content("Hello, world!")
        self.assertEqual(result, "Hello, world!")

    def test_list_with_single_text_item(self):
        content = [{"extras": {"signature": "EjQKC..."}, "index": 0, "text": "The answer"}]
        result = _normalize_content(content)
        self.assertEqual(result, "The answer")

    def test_list_with_multiple_text_items(self):
        content = [
            {"text": "Part one. ", "index": 0},
            {"text": "Part two.", "index": 1},
        ]
        result = _normalize_content(content)
        self.assertEqual(result, "Part one. Part two.")

    def test_list_items_without_text_key_are_skipped(self):
        content = [
            {"extras": {"signature": "abc"}, "index": 0},  # no 'text'
            {"text": "Real answer", "index": 1},
        ]
        result = _normalize_content(content)
        self.assertEqual(result, "Real answer")

    def test_empty_list_returns_empty_string(self):
        result = _normalize_content([])
        self.assertEqual(result, "")

    def test_non_dict_non_string_falls_back_to_str(self):
        result = _normalize_content(42)
        self.assertEqual(result, "42")


# ---------------------------------------------------------------------------
# Unit tests for _normalize_response
# ---------------------------------------------------------------------------


@unittest.skipIf(ModelResponse is None, "ModelResponse not importable")
class TestNormalizeResponse(unittest.TestCase):
    def _make_response(self, messages):
        return ModelResponse(result=messages, structured_response=None)

    def test_plain_string_ai_message_unchanged(self):
        msg = AIMessage(content="plain text")
        response = self._make_response([msg])
        result = _normalize_response(response)
        self.assertIs(result, response)  # same object — no copy needed
        self.assertEqual(result.result[0].content, "plain text")

    def test_list_content_is_normalized_to_string(self):
        gemini_content = [
            {"extras": {"signature": "EjQKC..."}, "index": 0, "text": "The actual answer here..."}
        ]
        msg = AIMessage(content=gemini_content)
        response = self._make_response([msg])
        result = _normalize_response(response)
        self.assertIsInstance(result.result[0].content, str)
        self.assertEqual(result.result[0].content, "The actual answer here...")

    def test_normalized_message_preserves_tool_calls(self):
        gemini_content = [{"text": "I'll use a tool", "index": 0}]
        tool_calls = [{"name": "search", "args": {}, "id": "tc1", "type": "tool_call"}]
        msg = AIMessage(content=gemini_content, tool_calls=tool_calls)
        response = self._make_response([msg])
        result = _normalize_response(response)
        self.assertEqual(result.result[0].tool_calls, tool_calls)

    def test_non_ai_messages_are_not_modified(self):
        human_msg = HumanMessage(content="hi")
        response = self._make_response([human_msg])
        result = _normalize_response(response)
        self.assertIs(result, response)
        self.assertIs(result.result[0], human_msg)

    def test_mixed_messages_only_ai_normalized(self):
        human_msg = HumanMessage(content="question")
        gemini_ai = AIMessage(
            content=[{"text": "answer", "index": 0, "extras": {"signature": "sig123"}}]
        )
        response = self._make_response([human_msg, gemini_ai])
        result = _normalize_response(response)
        # Human message untouched
        self.assertIs(result.result[0], human_msg)
        # AI message normalized
        self.assertIsInstance(result.result[1].content, str)
        self.assertEqual(result.result[1].content, "answer")

    def test_no_extras_signature_in_normalized_content(self):
        gemini_content = [
            {"extras": {"signature": "EjQKC_some_long_base64_string=="}, "index": 0, "text": "My answer"}
        ]
        msg = AIMessage(content=gemini_content)
        response = self._make_response([msg])
        result = _normalize_response(response)
        content = result.result[0].content
        self.assertNotIn("signature", content)
        self.assertNotIn("extras", content)


# ---------------------------------------------------------------------------
# Integration test: ModelRetryMiddleware normalizes on successful response
# ---------------------------------------------------------------------------


@unittest.skipIf(ModelResponse is None, "ModelResponse not importable")
class TestModelRetryMiddlewareNormalization(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_list_content_normalized_by_middleware(self):
        """Middleware must return plain string content even when Gemini returns a list."""
        gemini_content = [
            {"extras": {"signature": "EjQKC..."}, "index": 0, "text": "The docs answer"}
        ]
        ai_msg = AIMessage(content=gemini_content)
        fake_response = ModelResponse(result=[ai_msg], structured_response=None)

        async def fake_handler(request):
            return fake_response

        middleware = ModelRetryMiddleware(max_retries=0)
        request = MagicMock()  # content of request is irrelevant for this test

        result = await middleware.awrap_model_call(request, fake_handler)

        self.assertEqual(len(result.result), 1)
        ai_result = result.result[0]
        self.assertIsInstance(ai_result.content, str)
        self.assertEqual(ai_result.content, "The docs answer")

    async def test_plain_string_response_unchanged_by_middleware(self):
        """Non-Gemini plain string responses must pass through unmodified."""
        ai_msg = AIMessage(content="Normal answer from OpenAI/Anthropic")
        fake_response = ModelResponse(result=[ai_msg], structured_response=None)

        async def fake_handler(request):
            return fake_response

        middleware = ModelRetryMiddleware(max_retries=0)
        request = MagicMock()

        result = await middleware.awrap_model_call(request, fake_handler)

        self.assertEqual(result.result[0].content, "Normal answer from OpenAI/Anthropic")


if __name__ == "__main__":
    unittest.main()
