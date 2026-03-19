"""Unit tests for MessageNormalizationMiddleware.

Tests verify that non-consecutive SystemMessages are converted to HumanMessages
with a [Context: ...] prefix, while well-formed message lists pass through
unchanged.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.middleware.message_normalization import MessageNormalizationMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_middleware() -> MessageNormalizationMiddleware:
    return MessageNormalizationMiddleware()


# ---------------------------------------------------------------------------
# 1. _normalize_messages static method — no mock needed
# ---------------------------------------------------------------------------


class TestNormalizeMessages:
    """Direct tests for the static normalisation helper."""

    def test_no_system_messages_unchanged(self):
        """A conversation with no system messages is returned as-is."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
        ]
        result = MessageNormalizationMiddleware._normalize_messages(messages)
        assert result is messages  # same object — no copy made

    def test_leading_system_message_only_unchanged(self):
        """A single leading SystemMessage followed by human/AI is unchanged."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        result = MessageNormalizationMiddleware._normalize_messages(messages)
        assert result is messages

    def test_non_consecutive_system_message_converted(self):
        """SystemMessage after a HumanMessage is converted to HumanMessage."""
        messages = [
            SystemMessage(content="You are an assistant."),
            HumanMessage(content="first turn"),
            AIMessage(content="response"),
            SystemMessage(content="Context about the user's current page: PYTHON"),
            HumanMessage(content="second turn"),
        ]
        result = MessageNormalizationMiddleware._normalize_messages(messages)

        assert result is not messages
        assert len(result) == 5

        # Leading system message untouched
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are an assistant."

        # HumanMessage and AIMessage untouched
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

        # Non-consecutive system message converted
        assert isinstance(result[3], HumanMessage)
        assert result[3].content == "[Context: Context about the user's current page: PYTHON]"

        # Final human message untouched
        assert isinstance(result[4], HumanMessage)
        assert result[4].content == "second turn"

    def test_multiple_non_consecutive_system_messages_all_converted(self):
        """Multiple out-of-position SystemMessages are all converted."""
        messages = [
            SystemMessage(content="main prompt"),
            HumanMessage(content="turn 1"),
            AIMessage(content="answer 1"),
            SystemMessage(content="page context turn 2"),
            HumanMessage(content="turn 2"),
            AIMessage(content="answer 2"),
            SystemMessage(content="page context turn 3"),
            HumanMessage(content="turn 3"),
        ]
        result = MessageNormalizationMiddleware._normalize_messages(messages)

        assert result is not messages
        # Positions 3 and 6 (0-indexed) should be converted
        assert isinstance(result[3], HumanMessage)
        assert result[3].content == "[Context: page context turn 2]"
        assert isinstance(result[6], HumanMessage)
        assert result[6].content == "[Context: page context turn 3]"

        # All others retain their original type
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[4], HumanMessage)
        assert isinstance(result[5], AIMessage)
        assert isinstance(result[7], HumanMessage)

    def test_empty_messages_unchanged(self):
        """An empty message list is returned as-is."""
        messages: list = []
        result = MessageNormalizationMiddleware._normalize_messages(messages)
        assert result is messages

    def test_only_system_messages_unchanged(self):
        """A list composed entirely of SystemMessages (all consecutive) is unchanged."""
        messages = [
            SystemMessage(content="prompt 1"),
            SystemMessage(content="prompt 2"),
        ]
        result = MessageNormalizationMiddleware._normalize_messages(messages)
        assert result is messages

    def test_no_system_message_trace_match(self):
        """Reproduces the failing multi-turn trace structure from production."""
        messages = [
            SystemMessage(content="You are an expert LangChain customer service agent..."),
            SystemMessage(content="Context about the user's current page: viewing PYTHON..."),
            HumanMessage(content="react code"),
            AIMessage(content="I'd be happy to help..."),
            SystemMessage(content="Context about the user's current page: viewing PYTHON..."),
            HumanMessage(content="create_agent code"),
        ]
        result = MessageNormalizationMiddleware._normalize_messages(messages)

        # First two SystemMessages are before any non-system — both kept
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], SystemMessage)

        assert isinstance(result[2], HumanMessage)
        assert isinstance(result[3], AIMessage)

        # The 5th message (index 4) is a non-consecutive SystemMessage
        assert isinstance(result[4], HumanMessage)
        assert result[4].content.startswith("[Context: Context about the user's current page:")

        assert isinstance(result[5], HumanMessage)
        assert result[5].content == "create_agent code"


# ---------------------------------------------------------------------------
# 2. wrap_model_call (sync) — verifies middleware passes normalized request
# ---------------------------------------------------------------------------


class TestWrapModelCallSync:
    def test_passes_through_when_no_normalization_needed(self):
        mw = _make_middleware()
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="hi"),
        ]
        request = MagicMock()
        request.messages = messages
        handler = MagicMock(return_value=MagicMock())

        mw.wrap_model_call(request, handler)

        # handler called with the SAME request (no override needed)
        handler.assert_called_once_with(request)
        request.override.assert_not_called()

    def test_normalizes_and_overrides_request(self):
        mw = _make_middleware()
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="turn 1"),
            AIMessage(content="response"),
            SystemMessage(content="page context"),
            HumanMessage(content="turn 2"),
        ]
        normalized_request = MagicMock()
        request = MagicMock()
        request.messages = messages
        request.override.return_value = normalized_request
        handler = MagicMock(return_value=MagicMock())

        mw.wrap_model_call(request, handler)

        # override was called with normalized messages
        call_kwargs = request.override.call_args[1]
        new_msgs = call_kwargs["messages"]
        assert isinstance(new_msgs[3], HumanMessage)
        assert new_msgs[3].content == "[Context: page context]"

        # handler called with overridden request
        handler.assert_called_once_with(normalized_request)


# ---------------------------------------------------------------------------
# 3. awrap_model_call (async) — same checks asynchronously
# ---------------------------------------------------------------------------


class TestAwrapModelCallAsync:
    def test_passes_through_when_no_normalization_needed(self):
        async def _run():
            mw = _make_middleware()
            messages = [
                SystemMessage(content="system"),
                HumanMessage(content="hi"),
            ]
            request = MagicMock()
            request.messages = messages
            handler = AsyncMock(return_value=MagicMock())

            await mw.awrap_model_call(request, handler)

            handler.assert_called_once_with(request)
            request.override.assert_not_called()

        asyncio.run(_run())

    def test_normalizes_and_overrides_request(self):
        async def _run():
            mw = _make_middleware()
            messages = [
                SystemMessage(content="system"),
                HumanMessage(content="turn 1"),
                AIMessage(content="response"),
                SystemMessage(content="page context"),
                HumanMessage(content="turn 2"),
            ]
            normalized_request = MagicMock()
            request = MagicMock()
            request.messages = messages
            request.override.return_value = normalized_request
            handler = AsyncMock(return_value=MagicMock())

            await mw.awrap_model_call(request, handler)

            call_kwargs = request.override.call_args[1]
            new_msgs = call_kwargs["messages"]
            assert isinstance(new_msgs[3], HumanMessage)
            assert new_msgs[3].content == "[Context: page context]"

            handler.assert_called_once_with(normalized_request)

        asyncio.run(_run())
