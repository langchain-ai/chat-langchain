"""Tests for _fetch_all_articles() pagination in src/tools/pylon_tools.py.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

import importlib
import sys
import unittest
from unittest.mock import MagicMock, call, patch


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


ARTICLE_PAGE_1 = [
    {"id": "a1", "title": "Article 1"},
    {"id": "a2", "title": "Article 2"},
]
ARTICLE_PAGE_2 = [
    {"id": "a3", "title": "Article 3"},
    {"id": "a4", "title": "Article 4"},
]
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
    def test_cache_prevents_duplicate_requests(
        self, mock_get, mock_kb_id, mock_api_key
    ):
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


class TestSupportToolFailures(unittest.TestCase):
    def setUp(self):
        import src.tools.pylon_tools as pylon_module

        self.module = pylon_module
        self.article = {
            "id": "a1",
            "title": "Public article",
            "is_published": True,
            "visibility_config": {"visibility": "public"},
            "identifier": "123",
            "slug": "public-article",
            "collection_id": "c1",
            "current_published_content_html": "Verified content",
        }

    def tearDown(self):
        self.module._articles_cache = None
        self.module._collections_cache = None

    def invoke_tool(self, name):
        tool = getattr(self.module, name)
        return tool.invoke(
            {} if name == "search_support_articles" else {"article_id": "a1"}
        )

    def test_authorization_failures_are_explicit_for_both_endpoints_and_tools(self):
        import requests

        for name in ("search_support_articles", "get_support_article_content"):
            for endpoint in ("articles", "collections"):
                for status in (401, 403):
                    with self.subTest(tool=name, endpoint=endpoint, status=status):
                        response = requests.Response()
                        response.status_code = status
                        error = requests.HTTPError(
                            "sensitive-credential", response=response
                        )
                        self.module._articles_cache = None
                        self.module._collections_cache = None
                        failure = MagicMock()
                        failure.raise_for_status.side_effect = error
                        responses = (
                            [failure]
                            if endpoint == "articles"
                            else [_make_response([self.article]), failure]
                        )
                        with (
                            patch.object(self.module, "_get_headers", return_value={}),
                            patch.object(self.module, "_get_kb_id", return_value="kb"),
                            patch.object(
                                self.module.requests, "get", side_effect=responses
                            ),
                            self.assertLogs(self.module.logger, level="ERROR") as logs,
                        ):
                            result = self.invoke_tool(name)
                        self.assertTrue(result.startswith("SUPPORT_KB_UNAVAILABLE:"))
                        self.assertIn(
                            "Do not call support tools again this turn", result
                        )
                        self.assertIn("support, billing, or account", result)
                        self.assertIn(str(status), " ".join(logs.output))
                        self.assertNotIn(
                            "sensitive-credential", result + " ".join(logs.output)
                        )

    def test_successful_article_retrieval_is_preserved(self):
        with (
            patch.object(
                self.module, "_fetch_all_articles", return_value=[self.article]
            ),
            patch.object(
                self.module, "_fetch_collections", return_value={"General": "c1"}
            ),
        ):
            import json

            listing = json.loads(self.invoke_tool("search_support_articles"))
            content = self.invoke_tool("get_support_article_content")
        self.assertEqual(listing["articles"][0]["id"], "a1")
        self.assertIn("Collection: General", content)
        self.assertIn("Verified content", content)

    def test_unrelated_collection_failure_preserves_existing_fallbacks(self):
        import requests

        for name in ("search_support_articles", "get_support_article_content"):
            with (
                self.subTest(tool=name),
                patch.object(
                    self.module, "_fetch_all_articles", return_value=[self.article]
                ),
                patch.object(
                    self.module,
                    "_fetch_collections",
                    side_effect=requests.ConnectionError("offline"),
                ),
            ):
                result = self.invoke_tool(name)
            if name == "search_support_articles":
                self.assertIn("Failed to fetch collections: offline", result)
            else:
                self.assertIn("Collection: Customer Support Knowledge Base", result)
                self.assertIn("Verified content", result)


if __name__ == "__main__":
    unittest.main()
