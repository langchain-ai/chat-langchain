"""Tests for tool retry failure reporting."""

import unittest
from types import SimpleNamespace

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


class TestToolRetryMiddleware(unittest.IsolatedAsyncioTestCase):
    """Unit tests for tool failure message status."""

    def setUp(self):
        """Create middleware with retries disabled for fast tests."""
        self.middleware = ToolRetryMiddleware(max_attempts=1)
        self.request = SimpleNamespace(
            tool_call={"name": "search_support_articles", "id": "call-1"}
        )

    async def test_exhausted_failure_is_marked_as_error(self):
        """Exhausted tool failures remain visible as errored tool messages."""
        async def handler(request):
            raise RuntimeError("Pylon knowledge base unavailable: 401 Unauthorized")

        result = await self.middleware.awrap_tool_call(self.request, handler)

        self.assertEqual(result.status, "error")
        self.assertIn("Tool unavailable", result.content)

    def test_normalized_no_results_remains_success(self):
        """Normalized no-result responses remain ordinary tool output."""
        result = self.middleware._tool_message(self.request, "No results found.")

        self.assertEqual(result.status, "success")


if __name__ == "__main__":
    unittest.main()
