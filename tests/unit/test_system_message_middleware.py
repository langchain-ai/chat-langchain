"""Unit tests for SystemMessageMiddleware.

Guards against regressions where non-consecutive system messages in
multi-turn conversations cause Anthropic API to raise:
  ValueError('Received multiple non-consecutive system messages.')
"""

import asyncio
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(messages):
    """Build a minimal mock ModelRequest with the given messages."""
    request = MagicMock()
    request.messages = messages
    # model_copy returns a new mock with the updated messages
    def model_copy(*, update=None):
        new_req = MagicMock()
        new_req.messages = update.get("messages", messages) if update else messages
        new_req.model_copy = model_copy
        return new_req
    request.model_copy = model_copy
    return request


def _run(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSystemMessageMiddleware(unittest.TestCase):
    """Tests for SystemMessageMiddleware.awrap_model_call."""

    def setUp(self):
        from src.middleware.system_message_middleware import SystemMessageMiddleware
        self.middleware = SystemMessageMiddleware()

    # ------------------------------------------------------------------
    # Single system message — no merging needed
    # ------------------------------------------------------------------

    def test_single_system_message_passes_through_unchanged(self):
        """When there is only one system message, messages are unchanged."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
        ]
        request = _make_request(messages)

        captured_request = None

        async def handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        _run(self.middleware.awrap_model_call(request, handler))

        # The original request object should be passed through unmodified
        self.assertIs(captured_request, request)

    # ------------------------------------------------------------------
    # All system messages at the start — consecutive, no merging needed
    # ------------------------------------------------------------------

    def test_consecutive_system_messages_at_start_passes_through_unchanged(self):
        """Consecutive system messages at the start are not merged."""
        messages = [
            SystemMessage(content="System prompt."),
            SystemMessage(content="Page context."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        request = _make_request(messages)

        captured_request = None

        async def handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        _run(self.middleware.awrap_model_call(request, handler))

        # Both system messages are consecutive at the start — no merging
        self.assertIs(captured_request, request)

    # ------------------------------------------------------------------
    # Non-consecutive system messages — must be merged
    # ------------------------------------------------------------------

    def test_non_consecutive_system_messages_are_merged(self):
        """Non-consecutive system messages are merged into a single SystemMessage at index 0."""
        messages = [
            SystemMessage(content="Main system prompt."),
            SystemMessage(content="Page context turn 1."),
            HumanMessage(content="First question"),
            AIMessage(content="First answer"),
            SystemMessage(content="Page context turn 2."),
            HumanMessage(content="Second question"),
        ]
        request = _make_request(messages)

        captured_request = None

        async def handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        _run(self.middleware.awrap_model_call(request, handler))

        result_messages = captured_request.messages

        # There should be exactly one system message
        system_msgs = [m for m in result_messages if isinstance(m, SystemMessage)]
        self.assertEqual(len(system_msgs), 1, "Expected exactly one merged system message")

        # It should be at position 0
        self.assertIsInstance(result_messages[0], SystemMessage)

        # Its content should include all three original system message contents
        merged_content = result_messages[0].content
        self.assertIn("Main system prompt.", merged_content)
        self.assertIn("Page context turn 1.", merged_content)
        self.assertIn("Page context turn 2.", merged_content)

        # The non-system messages should be preserved in order
        non_system = [m for m in result_messages if not isinstance(m, SystemMessage)]
        self.assertEqual(len(non_system), 3)
        self.assertIsInstance(non_system[0], HumanMessage)
        self.assertEqual(non_system[0].content, "First question")
        self.assertIsInstance(non_system[1], AIMessage)
        self.assertEqual(non_system[1].content, "First answer")
        self.assertIsInstance(non_system[2], HumanMessage)
        self.assertEqual(non_system[2].content, "Second question")

    # ------------------------------------------------------------------
    # Three-turn conversation with interleaved system messages
    # ------------------------------------------------------------------

    def test_three_turn_conversation_merges_all_system_messages(self):
        """Simulates the exact failure pattern from production traces."""
        messages = [
            SystemMessage(content="You are an expert LangChain customer service agent..."),
            SystemMessage(content="Context about the user's current page: /python/..."),
            HumanMessage(content="First question"),
            AIMessage(content="First answer"),
            HumanMessage(content="Follow up"),
            AIMessage(content="Follow up answer"),
            SystemMessage(content="Context about the user's current page: /js/..."),
            HumanMessage(content="Second question"),
            AIMessage(content="Second answer"),
            SystemMessage(content="Context about the user's current page: /langgraph/..."),
            HumanMessage(content="Third question"),
        ]
        request = _make_request(messages)

        captured_request = None

        async def handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        _run(self.middleware.awrap_model_call(request, handler))

        result_messages = captured_request.messages

        # Should have exactly one system message
        system_msgs = [m for m in result_messages if isinstance(m, SystemMessage)]
        self.assertEqual(len(system_msgs), 1)

        # All four system message contents should be present in the merged message
        merged = result_messages[0].content
        self.assertIn("You are an expert LangChain customer service agent...", merged)
        self.assertIn("Context about the user's current page: /python/...", merged)
        self.assertIn("Context about the user's current page: /js/...", merged)
        self.assertIn("Context about the user's current page: /langgraph/...", merged)

        # Human and AI messages should be preserved
        non_system = [m for m in result_messages if not isinstance(m, SystemMessage)]
        self.assertEqual(len(non_system), 7)

    # ------------------------------------------------------------------
    # No system messages — should pass through unchanged
    # ------------------------------------------------------------------

    def test_no_system_messages_passes_through_unchanged(self):
        """When there are no system messages at all, messages are unchanged."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        request = _make_request(messages)

        captured_request = None

        async def handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        _run(self.middleware.awrap_model_call(request, handler))

        self.assertIs(captured_request, request)


if __name__ == "__main__":
    unittest.main()
