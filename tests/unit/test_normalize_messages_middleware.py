"""Tests for NormalizeInboundSystemMessagesMiddleware.

This middleware prevents ChatAnthropic from crashing with
``ValueError('Received multiple non-consecutive system messages.')`` when
the frontend injects a docs-page-context SystemMessage on top of the agent's
own system_prompt.

These tests do not require network access or LangSmith credentials.
"""

import unittest

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from src.middleware.normalize_messages_middleware import (
    NormalizeInboundSystemMessagesMiddleware,
)


class TestNormalizeInboundSystemMessagesMiddleware(unittest.TestCase):
    def setUp(self):
        self.mw = NormalizeInboundSystemMessagesMiddleware()

    def test_no_system_message_returns_none(self):
        """When there are no SystemMessages, the middleware is a no-op."""
        state = {"messages": [HumanMessage(content="hi")]}
        result = self.mw.before_agent(state, runtime=None)
        self.assertIsNone(result)

    def test_empty_messages_returns_none(self):
        state = {"messages": []}
        result = self.mw.before_agent(state, runtime=None)
        self.assertIsNone(result)

    def test_missing_messages_key_returns_none(self):
        state = {}
        result = self.mw.before_agent(state, runtime=None)
        self.assertIsNone(result)

    def test_system_message_is_converted_to_human(self):
        state = {
            "messages": [
                SystemMessage(content="Context about the user's current page..."),
                HumanMessage(content="how do I use create_agent?"),
            ]
        }
        result = self.mw.before_agent(state, runtime=None)
        self.assertIsNotNone(result)
        # First entry is the RemoveMessage sentinel, then the rewritten list.
        new_messages = result["messages"][1:]
        self.assertEqual(len(new_messages), 2)
        self.assertIsInstance(new_messages[0], HumanMessage)
        self.assertEqual(
            new_messages[0].content,
            "Context about the user's current page...",
        )
        self.assertIsInstance(new_messages[1], HumanMessage)
        self.assertEqual(new_messages[1].content, "how do I use create_agent?")

    def test_multiple_system_messages_all_converted(self):
        state = {
            "messages": [
                SystemMessage(content="ctx A"),
                HumanMessage(content="q1"),
                AIMessage(content="a1"),
                SystemMessage(content="ctx B"),
                HumanMessage(content="q2"),
            ]
        }
        result = self.mw.before_agent(state, runtime=None)
        self.assertIsNotNone(result)
        new_messages = result["messages"][1:]
        # None of the rewritten messages should be SystemMessage anymore.
        self.assertFalse(
            any(isinstance(m, SystemMessage) for m in new_messages),
            "middleware must convert every inbound SystemMessage",
        )
        self.assertEqual(len(new_messages), 5)
        # Order is preserved.
        self.assertEqual(new_messages[0].content, "ctx A")
        self.assertEqual(new_messages[3].content, "ctx B")

    def test_ai_and_human_messages_are_preserved(self):
        state = {
            "messages": [
                SystemMessage(content="ctx"),
                HumanMessage(content="q"),
                AIMessage(content="a"),
            ]
        }
        result = self.mw.before_agent(state, runtime=None)
        new_messages = result["messages"][1:]
        self.assertIsInstance(new_messages[1], HumanMessage)
        self.assertIsInstance(new_messages[2], AIMessage)


if __name__ == "__main__":
    unittest.main()
