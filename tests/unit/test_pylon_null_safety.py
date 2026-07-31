"""Tests for null-safe Pylon responses and hard-failure signalling.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

import unittest
from unittest.mock import MagicMock, patch

import src.tools.pylon_tools as pylon_module


def _make_raw_response(body):
    """Build a mock requests.Response whose .json() returns body verbatim."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestFetchAllArticlesNullSafety(unittest.TestCase):
    """_fetch_all_articles() must tolerate explicit nulls in the response body."""

    def setUp(self):
        pylon_module._articles_cache = None
        pylon_module._articles_failure_until = 0.0

    def tearDown(self):
        pylon_module._articles_cache = None
        pylon_module._articles_failure_until = 0.0

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_null_data_returns_empty_list(self, mock_get, mock_kb_id, mock_api_key):
        mock_get.return_value = _make_raw_response({"data": None})

        result = pylon_module._fetch_all_articles()

        self.assertEqual(result, [])

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_null_pagination_containers(self, mock_get, mock_kb_id, mock_api_key):
        articles = [{"id": "a1"}]
        mock_get.return_value = _make_raw_response(
            {"data": articles, "meta": None, "links": None, "pagination": None}
        )

        result = pylon_module._fetch_all_articles()

        self.assertEqual(result, articles)

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_failure_cooldown_short_circuits(self, mock_get, mock_kb_id, mock_api_key):
        pylon_module._mark_articles_unavailable()

        with self.assertRaises(pylon_module.SupportKBUnavailable):
            pylon_module._fetch_all_articles()

        mock_get.assert_not_called()


class TestSearchSupportArticlesHardFailure(unittest.TestCase):
    """Unexpected failures must raise, not come back as success-shaped content."""

    def setUp(self):
        pylon_module._articles_cache = None
        pylon_module._articles_failure_until = 0.0

    def tearDown(self):
        pylon_module._articles_cache = None
        pylon_module._articles_failure_until = 0.0

    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_unexpected_error_raises(self, mock_fetch):
        mock_fetch.side_effect = TypeError("'NoneType' object is not iterable")

        with self.assertRaises(pylon_module.SupportKBUnavailable) as ctx:
            pylon_module.search_support_articles.invoke({"collections": "all"})

        self.assertIn("TOOL ERROR:", str(ctx.exception))
        self.assertGreater(pylon_module._articles_failure_until, 0.0)

    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_get_article_content_unexpected_error_raises(self, mock_fetch):
        mock_fetch.side_effect = TypeError("'NoneType' object is not iterable")

        with self.assertRaises(pylon_module.SupportKBUnavailable) as ctx:
            pylon_module.get_support_article_content.invoke({"article_id": "a1"})

        self.assertIn("TOOL ERROR:", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
