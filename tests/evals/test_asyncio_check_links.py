"""Tests verifying that check_links works correctly in async contexts.

The bug: check_links internally calls asyncio.run(_check_urls_async(...)).
When invoked from within an already-running event loop (e.g., LangGraph's
ToolNode calls sync tools synchronously inside an async task), Python raises:
    RuntimeError: This event loop is already running
"""

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.link_check_tools import LinkCheckResult, _run_async, check_links


# ---------------------------------------------------------------------------
# Helper to simulate LangGraph calling check_links from within a running loop.
# LangGraph's ToolNode calls tool.invoke() directly inside an async coroutine.
# We replicate this by running check_links inside an async coroutine via
# run_until_complete in a dedicated thread (so the thread has a running loop).
# ---------------------------------------------------------------------------


def _invoke_check_links_in_running_loop(urls: list[str]) -> str:
    """
    Simulate LangGraph calling check_links synchronously from inside an async
    task (not via run_in_executor).  LangGraph's ToolNode calls tool.invoke()
    directly from an async function body, meaning the event loop IS running when
    check_links executes its internal asyncio.run() call.

    We replicate this by running check_links inside an async coroutine in a
    dedicated thread where loop.is_running() is True.
    """
    result_holder: dict = {}
    error_holder: dict = {}

    def _thread_with_running_loop():
        """Thread that sets up a loop and calls check_links from inside a coroutine."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _async_caller():
            # loop.is_running() is True here.
            # call check_links synchronously (as LangGraph ToolNode does).
            # Without the fix, check_links raises RuntimeError here.
            try:
                result = check_links.invoke({"urls": urls})
                result_holder["result"] = result
            except RuntimeError as e:
                error_holder["error"] = str(e)

        try:
            loop.run_until_complete(_async_caller())
        finally:
            loop.close()

    t = threading.Thread(target=_thread_with_running_loop)
    t.start()
    t.join()

    if "error" in error_holder:
        raise RuntimeError(error_holder["error"])
    return result_holder.get("result", "")


# ---------------------------------------------------------------------------
# Tests for the _run_async helper
# ---------------------------------------------------------------------------


class TestRunAsyncHelper:
    """Unit tests for the _run_async helper function."""

    def test_run_async_works_without_running_loop(self):
        """_run_async works correctly when no event loop is running."""

        async def _coro():
            return 42

        result = _run_async(_coro())
        assert result == 42

    def test_asyncio_run_raises_in_already_running_loop(self):
        """
        Baseline: confirm that vanilla asyncio.run() raises RuntimeError when
        a loop is already running — this is the root-cause bug check_links had.
        """

        async def _inner():
            async def _noop():
                return 1

            with pytest.raises(RuntimeError, match="This event loop is already running"):
                asyncio.run(_noop())

        asyncio.run(_inner())


# ---------------------------------------------------------------------------
# Tests for check_links called from within an async context
# ---------------------------------------------------------------------------


class TestCheckLinksInAsyncContext:
    """Tests that check_links works when called from within an async event loop."""

    def _make_mock_result(self, url: str, valid: bool = True) -> LinkCheckResult:
        return LinkCheckResult(
            url=url,
            valid=valid,
            status_code=200 if valid else 404,
            error=None if valid else "HTTP 404",
        )

    def test_check_links_does_not_raise_runtime_error_in_async_context(self):
        """
        Regression test: check_links must NOT raise RuntimeError when called
        from within a running event loop (the exact failure mode in LangGraph).

        Without the fix: asyncio.run() inside check_links raises
        RuntimeError('This event loop is already running').
        """
        mock_results = [self._make_mock_result("https://example.com", valid=True)]

        with patch(
            "src.tools.link_check_tools._check_urls_async",
            new=AsyncMock(return_value=mock_results),
        ):
            raised = None
            try:
                result = _invoke_check_links_in_running_loop(["https://example.com"])
            except RuntimeError as e:
                if "This event loop is already running" in str(e):
                    raised = e
                else:
                    raise

        assert raised is None, (
            f"check_links raised RuntimeError('This event loop is already running') "
            f"when called from an async context. This is the asyncio bug.\n"
            f"Error: {raised}"
        )

    def test_check_links_returns_correct_result_in_async_context(self):
        """check_links should return properly formatted results when called from async context."""
        mock_results = [self._make_mock_result("https://docs.langchain.com/", valid=True)]

        with patch(
            "src.tools.link_check_tools._check_urls_async",
            new=AsyncMock(return_value=mock_results),
        ):
            result = _invoke_check_links_in_running_loop(["https://docs.langchain.com/"])

        assert isinstance(result, str)
        assert "1/1 valid" in result

    def test_check_links_in_sync_context(self):
        """check_links should still work correctly in a plain sync context."""
        mock_results = [self._make_mock_result("https://example.com", valid=True)]

        with patch(
            "src.tools.link_check_tools._check_urls_async",
            new=AsyncMock(return_value=mock_results),
        ):
            result = check_links.invoke({"urls": ["https://example.com"]})

        assert isinstance(result, str)
        assert "1/1 valid" in result

    def test_check_links_empty_urls(self):
        """check_links with empty list returns early without running async code."""
        result = check_links.invoke({"urls": []})
        assert result == "No URLs provided to check."

    def test_check_links_deduplicates_urls(self):
        """check_links deduplicates URLs before checking."""
        mock_results = [self._make_mock_result("https://example.com", valid=True)]

        with patch(
            "src.tools.link_check_tools._check_urls_async",
            new=AsyncMock(return_value=mock_results),
        ) as mock_async:
            check_links.invoke({"urls": ["https://example.com", "https://example.com"]})

        # Should only check once despite duplicate input
        call_args = mock_async.call_args
        checked_urls = call_args[0][0]
        assert len(checked_urls) == 1
        assert checked_urls[0] == "https://example.com"

    def test_check_links_handles_invalid_url_in_async_context(self):
        """check_links should report invalid URLs correctly when called from async context."""
        mock_results = [
            self._make_mock_result("https://example.com", valid=True),
            LinkCheckResult(url="not-a-url", valid=False, error="Invalid URL format"),
        ]

        with patch(
            "src.tools.link_check_tools._check_urls_async",
            new=AsyncMock(return_value=mock_results),
        ):
            result = _invoke_check_links_in_running_loop(
                ["https://example.com", "not-a-url"]
            )

        assert "1/2 valid" in result
        assert "not-a-url" in result

    def test_check_links_called_multiple_times_from_async_context(self):
        """check_links can be called multiple times from async context without error."""
        mock_results = [self._make_mock_result("https://example.com", valid=True)]

        errors = []
        results = []

        def run_one():
            try:
                with patch(
                    "src.tools.link_check_tools._check_urls_async",
                    new=AsyncMock(return_value=mock_results),
                ):
                    r = _invoke_check_links_in_running_loop(["https://example.com"])
                    results.append(r)
            except RuntimeError as e:
                errors.append(str(e))

        # Run sequentially (multiple calls as LangGraph would over a conversation)
        for _ in range(3):
            run_one()

        assert len(errors) == 0, f"Got RuntimeErrors: {errors}"
        assert len(results) == 3
