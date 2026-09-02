"""Tests for _fetch_all_articles() pagination in src/tools/pylon_tools.py.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests
from langchain_core.tools import ToolException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(data, next_cursor=None):
    """Build a mock requests.Response whose .json() returns a Pylon-shaped body."""
    body = {"data": data}
    if next_cursor is not None:
        body["next"] = next_cursor
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _make_meta_response(data, next_cursor=None):
    """Build a mock response that uses the meta.next pagination shape."""
    body = {"data": data, "meta": {}}
    if next_cursor is not None:
        body["meta"]["next"] = next_cursor
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status.return_value = None
    return mock_resp


ARTICLE_PAGE_1 = [{"id": "a1", "title": "Article 1"}, {"id": "a2", "title": "Article 2"}]
ARTICLE_PAGE_2 = [{"id": "a3", "title": "Article 3"}, {"id": "a4", "title": "Article 4"}]
ARTICLE_PAGE_3 = [{"id": "a5", "title": "Article 5"}]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFetchAllArticlesPagination(unittest.TestCase):
    """Unit tests for _fetch_all_articles() pagination behaviour."""

    def setUp(self):
        """Reload the module and reset the global cache before each test."""
        # Ensure a clean module state so _articles_cache starts as None
        import src.tools.pylon_tools as pylon_module

        pylon_module._articles_cache = None
        pylon_module._collections_cache = None
        self.module = pylon_module

    # ------------------------------------------------------------------
    # Single-page case (no "next" key -> stops after one request)
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_single_page_no_next(self, mock_get, mock_kb_id, mock_api_key):
        """When no 'next' cursor is present, exactly one HTTP request is made."""
        mock_get.return_value = _make_response(ARTICLE_PAGE_1, next_cursor=None)

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1)
        mock_get.assert_called_once()

    # ------------------------------------------------------------------
    # Multi-page case (two pages)
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_two_pages_collected(self, mock_get, mock_kb_id, mock_api_key):
        """Articles from both pages are combined into a single list."""
        mock_get.side_effect = [
            _make_response(ARTICLE_PAGE_1, next_cursor="cursor-abc"),
            _make_response(ARTICLE_PAGE_2, next_cursor=None),
        ]

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1 + ARTICLE_PAGE_2)
        self.assertEqual(mock_get.call_count, 2)

        # Second call must pass cursor as query parameter
        _, kwargs = mock_get.call_args_list[1]
        self.assertEqual(kwargs.get("params", {}).get("cursor"), "cursor-abc")

    # ------------------------------------------------------------------
    # Three-page case
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_three_pages_collected(self, mock_get, mock_kb_id, mock_api_key):
        """Articles from all three pages are combined."""
        mock_get.side_effect = [
            _make_response(ARTICLE_PAGE_1, next_cursor="cursor-1"),
            _make_response(ARTICLE_PAGE_2, next_cursor="cursor-2"),
            _make_response(ARTICLE_PAGE_3, next_cursor=None),
        ]

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1 + ARTICLE_PAGE_2 + ARTICLE_PAGE_3)
        self.assertEqual(mock_get.call_count, 3)

    # ------------------------------------------------------------------
    # meta.next pagination shape
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_meta_next_pagination(self, mock_get, mock_kb_id, mock_api_key):
        """Handles responses that use meta.next instead of top-level next."""
        mock_get.side_effect = [
            _make_meta_response(ARTICLE_PAGE_1, next_cursor="meta-cursor-1"),
            _make_meta_response(ARTICLE_PAGE_2, next_cursor=None),
        ]

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1 + ARTICLE_PAGE_2)
        self.assertEqual(mock_get.call_count, 2)

    # ------------------------------------------------------------------
    # Safety limit: stops after max_pages (10) even if more pages exist
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_safety_limit_stops_at_max_pages(self, mock_get, mock_kb_id, mock_api_key):
        """The loop stops after 10 pages even if the API keeps returning next cursors."""
        # Every page claims there is another page after it
        mock_get.side_effect = [
            _make_response([{"id": f"a{i}"}], next_cursor=f"cursor-{i}")
            for i in range(20)  # more than the 10-page cap
        ]

        result = self.module._fetch_all_articles()

        # Should have fetched exactly 10 articles (one per page, capped at 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(mock_get.call_count, 10)

    # ------------------------------------------------------------------
    # Cache: second call returns cached result without extra HTTP requests
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_cache_prevents_duplicate_requests(self, mock_get, mock_kb_id, mock_api_key):
        """Calling _fetch_all_articles() twice only hits the network once."""
        mock_get.return_value = _make_response(ARTICLE_PAGE_1, next_cursor=None)

        first = self.module._fetch_all_articles()
        second = self.module._fetch_all_articles()

        self.assertEqual(first, second)
        mock_get.assert_called_once()

    # ------------------------------------------------------------------
    # Empty data list (no articles at all)
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_empty_data_returns_empty_list(self, mock_get, mock_kb_id, mock_api_key):
        """An API response with no articles returns an empty list."""
        mock_get.return_value = _make_response([], next_cursor=None)

        result = self.module._fetch_all_articles()

        self.assertEqual(result, [])
        mock_get.assert_called_once()

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_null_data_returns_empty_list(self, mock_get, mock_kb_id, mock_api_key):
        """An explicit null article data value is treated as an empty list."""
        mock_get.return_value = _make_response(None, next_cursor=None)

        result = self.module._fetch_all_articles()

        self.assertEqual(result, [])

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_null_collection_data_returns_empty_mapping(
        self, mock_get, mock_kb_id, mock_api_key
    ):
        """An explicit null collection data value is treated as an empty list."""
        mock_get.return_value = _make_response(None, next_cursor=None)

        result = self.module._fetch_collections()

        self.assertEqual(result, {})


class TestPylonToolErrors(unittest.TestCase):
    """Unit tests for Pylon tool failure propagation."""

    def setUp(self):
        """Reset the global caches before each test."""
        import src.tools.pylon_tools as pylon_module

        pylon_module._articles_cache = None
        pylon_module._collections_cache = None
        self.module = pylon_module

    def test_search_unexpected_error_raises_tool_exception(self):
        """Unexpected search failures are exposed as tool errors."""
        with patch.object(
            self.module, "_fetch_all_articles", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(ToolException, "Unexpected error: boom"):
                self.module.search_support_articles.invoke({"collections": "all"})

    def test_search_configuration_error_raises_tool_exception(self):
        """Search configuration failures are exposed as tool errors."""
        with patch.object(
            self.module,
            "_fetch_all_articles",
            side_effect=ValueError("PYLON_API_KEY not configured"),
        ):
            with self.assertRaisesRegex(ToolException, "PYLON_API_KEY not configured"):
                self.module.search_support_articles.invoke({"collections": "all"})

    def test_search_request_error_raises_tool_exception(self):
        """Search request failures are exposed as tool errors."""
        with patch.object(
            self.module,
            "_fetch_all_articles",
            side_effect=requests.exceptions.RequestException("network down"),
        ):
            with self.assertRaisesRegex(ToolException, "network down"):
                self.module.search_support_articles.invoke({"collections": "all"})

    def test_content_unexpected_error_raises_tool_exception(self):
        """Unexpected content failures are exposed as tool errors."""
        with patch.object(
            self.module, "_fetch_all_articles", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(ToolException, "Unexpected error: boom"):
                self.module.get_support_article_content.invoke({"article_id": "a1"})

    def test_content_configuration_error_raises_tool_exception(self):
        """Content configuration failures are exposed as tool errors."""
        with patch.object(
            self.module,
            "_fetch_all_articles",
            side_effect=ValueError("PYLON_API_KEY not configured"),
        ):
            with self.assertRaisesRegex(ToolException, "PYLON_API_KEY not configured"):
                self.module.get_support_article_content.invoke({"article_id": "a1"})

    def test_content_request_error_raises_tool_exception(self):
        """Content request failures are exposed as tool errors."""
        with patch.object(
            self.module,
            "_fetch_all_articles",
            side_effect=requests.exceptions.RequestException("network down"),
        ):
            with self.assertRaisesRegex(
                ToolException, "Error fetching article: network down"
            ):
                self.module.get_support_article_content.invoke({"article_id": "a1"})


if __name__ == "__main__":
    unittest.main()
