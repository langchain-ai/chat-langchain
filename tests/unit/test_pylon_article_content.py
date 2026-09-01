"""Tests for support article identifier resolution."""

import json
import unittest
from unittest.mock import patch

ARTICLE = {
    "id": "ABC-123",
    "identifier": 456,
    "slug": "getting-started",
    "title": "Getting Started",
    "collection_id": "collection-1",
    "current_published_content_html": "<p>Article content</p>",
}


class TestGetSupportArticleContent(unittest.TestCase):
    """Unit tests for support article identifier resolution."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_resolves_by_uuid(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": " abc-123 "})

        self.assertIn("<p>Article content</p>", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_resolves_by_numeric_identifier(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": "456"})

        self.assertIn("<p>Article content</p>", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_resolves_by_identifier_slug(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke(
            {"article_id": "456-GETTING-STARTED"}
        )

        self.assertIn("<p>Article content</p>", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_unknown_value_returns_error_envelope(
        self, mock_articles, mock_collections
    ):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": "missing"})

        self.assertEqual(
            json.loads(result),
            {
                "error": "No support article matches 'missing'.",
                "hint": "Pass the UUID from the 'id' field of search_support_articles.",
            },
        )


if __name__ == "__main__":
    unittest.main()
