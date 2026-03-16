"""Unit tests for context-overflow handling.

Covers:
1. retry_middleware skips retries on context-overflow errors (BadRequestError
   "prompt is too long") and re-raises immediately.
2. docs_tools._format_search_results truncates long content before inserting
   it into the LLM context.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _is_context_overflow,
)
from src.tools.docs_tools import MAX_CONTENT_CHARS_PER_RESULT, _format_search_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(content: str, path: str = "oss/python/langchain/tools", title: str = "Test") -> dict:
    return {"path": path, "metadata": {"title": title}, "content": content}


# ---------------------------------------------------------------------------
# _is_context_overflow
# ---------------------------------------------------------------------------


class TestIsContextOverflow:
    """Unit tests for the _is_context_overflow helper."""

    def test_prompt_is_too_long(self):
        assert _is_context_overflow(Exception("BadRequestError: prompt is too long (217K > 200K)"))

    def test_prompt_too_long_underscore(self):
        assert _is_context_overflow(Exception("error: prompt_too_long"))

    def test_context_length_exceeded(self):
        assert _is_context_overflow(Exception("context_length_exceeded: reduce prompt"))

    def test_maximum_context_length(self):
        assert _is_context_overflow(Exception("This exceeds the maximum context length"))

    def test_too_many_tokens(self):
        assert _is_context_overflow(Exception("too many tokens in the request"))

    def test_reduce_the_length(self):
        assert _is_context_overflow(Exception("Please reduce the length of the messages"))

    def test_case_insensitive(self):
        assert _is_context_overflow(Exception("PROMPT IS TOO LONG"))

    def test_unrelated_error(self):
        assert not _is_context_overflow(Exception("connection timeout"))

    def test_rate_limit_not_overflow(self):
        assert not _is_context_overflow(Exception("RateLimitError: Too many requests"))

    def test_authentication_error_not_overflow(self):
        assert not _is_context_overflow(Exception("AuthenticationError: invalid API key"))


# ---------------------------------------------------------------------------
# ModelRetryMiddleware — context overflow skips retries
# ---------------------------------------------------------------------------


class TestRetryMiddlewareContextOverflow:
    """Verify that ModelRetryMiddleware does not retry context-overflow errors."""

    def test_context_overflow_raises_immediately_no_sleep(self):
        """A BadRequestError with 'prompt is too long' must not trigger any retry sleep."""
        middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0.5)
        overflow_exc = Exception("BadRequestError: prompt is too long (217K > 200K)")

        call_count = 0

        async def handler(_request):
            nonlocal call_count
            call_count += 1
            raise overflow_exc

        async def run():
            with patch("src.middleware.retry_middleware.asyncio.sleep") as mock_sleep:
                with pytest.raises(Exception, match="prompt is too long"):
                    await middleware.awrap_model_call(MagicMock(), handler)
                mock_sleep.assert_not_called()

        asyncio.get_event_loop().run_until_complete(run())
        # Should only be called once — no retries attempted
        assert call_count == 1, f"Expected 1 call, got {call_count} (retries occurred)"

    def test_transient_error_is_retried(self):
        """A transient error (e.g., connection timeout) must still trigger retries."""
        middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0.01)
        transient_exc = Exception("connection timeout")
        success_response = MagicMock(response_metadata={})

        call_count = 0

        async def handler(_request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise transient_exc
            return success_response

        async def run():
            with patch("src.middleware.retry_middleware.asyncio.sleep"):
                result = await middleware.awrap_model_call(MagicMock(), handler)
            assert result is success_response

        asyncio.get_event_loop().run_until_complete(run())
        assert call_count == 3, f"Expected 3 calls (2 failures + 1 success), got {call_count}"

    def test_context_overflow_not_retried_even_with_max_retries_3(self):
        """With max_retries=3, overflow errors must still short-circuit after 1 call."""
        middleware = ModelRetryMiddleware(max_retries=3, initial_delay=0.5)

        call_count = 0

        async def handler(_request):
            nonlocal call_count
            call_count += 1
            raise Exception("context_length_exceeded")

        async def run():
            with patch("src.middleware.retry_middleware.asyncio.sleep") as mock_sleep:
                with pytest.raises(Exception, match="context_length_exceeded"):
                    await middleware.awrap_model_call(MagicMock(), handler)
                mock_sleep.assert_not_called()

        asyncio.get_event_loop().run_until_complete(run())
        assert call_count == 1


# ---------------------------------------------------------------------------
# _format_search_results — content truncation
# ---------------------------------------------------------------------------


class TestFormatSearchResultsContentTruncation:
    """Verify that _format_search_results truncates long content."""

    def test_content_within_limit_is_not_truncated(self):
        """Short content must pass through unchanged."""
        short_content = "a" * (MAX_CONTENT_CHARS_PER_RESULT - 1)
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(short_content)])
        assert "... [truncated]" not in output
        assert short_content in output

    def test_content_at_exact_limit_is_not_truncated(self):
        """Content exactly at the limit must not be truncated."""
        exact_content = "b" * MAX_CONTENT_CHARS_PER_RESULT
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(exact_content)])
        assert "... [truncated]" not in output

    def test_content_over_limit_is_truncated(self):
        """Content exceeding the limit must be cut and marked with '[truncated]'."""
        long_content = "c" * (MAX_CONTENT_CHARS_PER_RESULT + 500)
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(long_content)])
        assert "... [truncated]" in output

    def test_truncated_content_does_not_exceed_limit_plus_marker(self):
        """The total content in output must not exceed the limit + len('... [truncated]')."""
        marker = "... [truncated]"
        long_content = "d" * (MAX_CONTENT_CHARS_PER_RESULT * 3)
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(long_content)])
        # Extract the content section from the output
        for line in output.split("\n"):
            if line.startswith("Content:"):
                content_value = line[len("Content:"):].strip()
                assert len(content_value) <= MAX_CONTENT_CHARS_PER_RESULT + len(marker)
                break

    def test_multiple_results_all_truncated(self):
        """All results in a multi-result response must be individually truncated."""
        long_content = "e" * (MAX_CONTENT_CHARS_PER_RESULT * 2)
        results = [
            _make_result(long_content, path=f"path/{i}", title=f"Doc {i}")
            for i in range(3)
        ]
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results(results)
        # Three truncation markers expected — one per result
        assert output.count("... [truncated]") == 3

    def test_default_max_content_chars_is_reasonable(self):
        """The default limit must be >= 1000 and <= 10000 to balance detail vs token cost."""
        assert 1000 <= MAX_CONTENT_CHARS_PER_RESULT <= 10000, (
            f"MAX_CONTENT_CHARS_PER_RESULT={MAX_CONTENT_CHARS_PER_RESULT} is outside "
            "the acceptable range [1000, 10000]."
        )
