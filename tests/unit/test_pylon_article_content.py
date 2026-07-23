"""Tests for get_support_article_content id resolution in src/tools/pylon_tools.py.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

import unittest
from unittest.mock import MagicMock, patch


def _make_list_response(data, next_cursor=None):
    """Build a mock articles-list response (Pylon-shaped body)."""
    body = {"data": data}
    if next_cursor is not None:
        body["next"] = next_cursor
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status.return_value = None
    mock_resp.status_code = 200
    return mock_resp


def _make_single_response(article):
    """Build a mock single-article response, or a 404 when article is None."""
    mock_resp = MagicMock()
    if article is None:
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"data": None}
    else:
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": article}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# A UUID that search surfaced but that lands beyond the paginated cache cap.
SURFACED_UUID = "123e4567-e89b-12d3-a456-426614174000"


class TestArticleContentResolution(unittest.TestCase):
    """Unit tests for get_support_article_content id resolution."""

    def setUp(self):
        import src.tools.pylon_tools as pylon_module

        pylon_module._articles_cache = None
        pylon_module._collections_cache = None
        self.module = pylon_module

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_uuid_beyond_cache_resolves_via_direct_fetch(
        self, mock_get, mock_kb_id, mock_api_key, mock_collections
    ):
        """A UUID search surfaced but missing from the cache resolves via direct fetch."""
        # The paginated cache does not contain SURFACED_UUID.
        cache_page = _make_list_response(
            [{"id": "other-id", "title": "Other"}], next_cursor=None
        )
        # The direct-by-id endpoint returns the article.
        single = _make_single_response(
            {
                "id": SURFACED_UUID,
                "title": "Deep Article",
                "identifier": "999",
                "slug": "deep-article",
                "current_published_content_html": "<p>hello</p>",
            }
        )
        mock_get.side_effect = [cache_page, single]

        result = self.module.get_support_article_content.invoke(
            {"article_id": SURFACED_UUID}
        )

        self.assertIn(SURFACED_UUID, result)
        self.assertIn("Deep Article", result)
        self.assertIn("hello", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_genuinely_missing_id_returns_actionable_hint(
        self, mock_get, mock_kb_id, mock_api_key, mock_collections
    ):
        """A truly absent id yields a not-found message that hints to re-run search."""
        cache_page = _make_list_response(
            [{"id": "other-id", "title": "Other"}], next_cursor=None
        )
        missing = _make_single_response(None)
        mock_get.side_effect = [cache_page, missing]

        result = self.module.get_support_article_content.invoke(
            {"article_id": SURFACED_UUID}
        )

        self.assertIn("not found", result)
        self.assertIn("search_support_articles", result)


if __name__ == "__main__":
    unittest.main()
