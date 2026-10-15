"""Tests that Pylon rate-limit failures propagate instead of becoming tool results.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests


def _make_response(data, next_cursor=None):
    """Build a mock requests.Response whose .json() returns a Pylon-shaped body."""
    body = {"data": data}
    if next_cursor is not None:
        body["next"] = next_cursor
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _make_429_error():
    """Build an HTTPError carrying a 429 response, as raise_for_status() would."""
    response = MagicMock()
    response.status_code = 429
    return requests.exceptions.HTTPError(
        "429 Client Error: Too Many Requests for url: "
        "https://api.usepylon.com/knowledge-bases/kb-123/articles",
        response=response,
    )


ARTICLE_PAGE_1 = [{"id": "a1", "title": "Article 1"}, {"id": "a2", "title": "Article 2"}]
ARTICLE_PAGE_2 = [{"id": "a3", "title": "Article 3"}, {"id": "a4", "title": "Article 4"}]


class TestPylonRateLimitHandling(unittest.TestCase):
    """Unit tests for 429 handling in src/tools/pylon_tools.py."""

    def setUp(self):
        """Reset module-level caches and backoff state before each test."""
        import src.tools.pylon_tools as pylon_module
        pylon_module._articles_cache = None
        pylon_module._collections_cache = None
        pylon_module._articles_backoff_until = 0.0
        self.module = pylon_module

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools._session_get")
    def test_429_raises_instead_of_returning_error_string(
        self, mock_get, mock_kb_id, mock_api_key
    ):
        """A 429 from the articles endpoint propagates so retry middleware can fire."""
        mock_get.side_effect = _make_429_error()

        with self.assertRaises(requests.exceptions.RequestException) as ctx:
            self.module.search_support_articles.invoke({"collections": "all"})

        self.assertEqual(ctx.exception.response.status_code, 429)

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools._session_get")
    def test_429_on_later_page_keeps_earlier_pages(
        self, mock_get, mock_kb_id, mock_api_key
    ):
        """A 429 on page three keeps the first two pages instead of discarding them."""
        mock_get.side_effect = [
            _make_response(ARTICLE_PAGE_1, next_cursor="cursor-1"),
            _make_response(ARTICLE_PAGE_2, next_cursor="cursor-2"),
            _make_429_error(),
        ]

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1 + ARTICLE_PAGE_2)
        self.assertEqual(self.module._articles_cache, ARTICLE_PAGE_1 + ARTICLE_PAGE_2)
        self.assertEqual(mock_get.call_count, 3)

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools._session_get")
    def test_429_without_partial_results_sets_backoff(
        self, mock_get, mock_kb_id, mock_api_key
    ):
        """After a 429 with no pages fetched, the next call backs off without new requests."""
        mock_get.side_effect = _make_429_error()

        with self.assertRaises(requests.exceptions.RequestException):
            self.module._fetch_all_articles()

        self.assertEqual(mock_get.call_count, 1)

        with self.assertRaises(requests.exceptions.RetryError):
            self.module._fetch_all_articles()

        self.assertEqual(mock_get.call_count, 1)

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools._session_get")
    def test_404_still_returned_as_tool_error(self, mock_get, mock_kb_id, mock_api_key):
        """Non-retryable client errors keep the existing JSON error behaviour."""
        response = MagicMock()
        response.status_code = 404
        mock_get.side_effect = requests.exceptions.HTTPError(
            "404 Client Error: Not Found", response=response
        )

        result = self.module.search_support_articles.invoke({"collections": "all"})

        self.assertIn("404", result)


if __name__ == "__main__":
    unittest.main()
