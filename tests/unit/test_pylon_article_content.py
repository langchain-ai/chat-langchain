"""Tests for support article content resolution."""

import unittest
from unittest.mock import patch

import src.tools.pylon_tools as pylon_module

ARTICLES = [
    {
        "id": "UUID-123",
        "identifier": "1242226068",
        "slug": "configure-billing",
        "title": "Configure billing",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>Article content</p>",
    }
]


class TestSupportArticleContentResolution(unittest.TestCase):
    """Unit tests for support article identifier resolution."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_resolves_by_uuid_id(self, mock_articles, mock_collections):
        result = pylon_module.get_support_article_content.invoke({"article_id": " uuid-123 "})

        self.assertIn("ID: UUID-123", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_resolves_by_bare_identifier(self, mock_articles, mock_collections):
        result = pylon_module.get_support_article_content.invoke({"article_id": "1242226068"})

        self.assertIn("ID: UUID-123", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_resolves_by_identifier_slug(self, mock_articles, mock_collections):
        result = pylon_module.get_support_article_content.invoke(
            {"article_id": "1242226068-configure-billing"}
        )

        self.assertIn("ID: UUID-123", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_unknown_id_returns_actionable_error(self, mock_articles, mock_collections):
        result = pylon_module.get_support_article_content.invoke({"article_id": "unknown-id"})

        self.assertIn("was not recognized", result)
        self.assertIn("UUIDs in the id field returned by search_support_articles", result)
        self.assertIn("Call search_support_articles for the relevant collections", result)
        self.assertNotIn("Article content", result)


if __name__ == "__main__":
    unittest.main()
