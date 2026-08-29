"""Tests for support article reference resolution."""

import unittest
from unittest.mock import patch

import src.tools.pylon_tools as pylon_module

ARTICLE = {
    "id": "uuid-123",
    "identifier": 456,
    "slug": "reset-your-password",
    "title": "Reset your password",
    "collection_id": "collection-1",
    "is_published": True,
    "visibility_config": {"visibility": "public"},
    "current_published_content_html": "<p>Reset steps</p>",
}


class TestGetSupportArticleContent(unittest.TestCase):
    """Unit tests for support article reference resolution."""

    def setUp(self):
        pylon_module._articles_cache = [ARTICLE]

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    def test_resolves_by_uuid_id(self, mock_fetch_collections):
        result = pylon_module.get_support_article_content.func("uuid-123")

        self.assertIn("Reset your password", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    def test_resolves_by_numeric_identifier(self, mock_fetch_collections):
        result = pylon_module.get_support_article_content.func("456")

        self.assertIn("Reset your password", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    def test_resolves_by_support_article_url(self, mock_fetch_collections):
        result = pylon_module.get_support_article_content.func(
            "https://support.langchain.com/articles/456-reset-your-password"
        )

        self.assertIn("Reset your password", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    def test_not_found_lists_valid_article_ids(self, mock_fetch_collections):
        result = pylon_module.get_support_article_content.func("missing")

        self.assertIn("Error: Article lookup failed", result)
        self.assertIn("uuid-123", result)
        self.assertIn("Retry with one of those ids", result)


if __name__ == "__main__":
    unittest.main()
