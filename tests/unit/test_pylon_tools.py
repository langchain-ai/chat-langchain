"""Tests for Pylon knowledge base tools."""

import unittest
from unittest.mock import patch


class TestGetSupportArticleContent(unittest.TestCase):
    """Unit tests for support article lookup."""

    def setUp(self):
        """Reset the Pylon article cache before each test."""
        import src.tools.pylon_tools as pylon_module

        self.module = pylon_module
        self.module._articles_cache = None
        self.module._collections_cache = None

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_lookup_accepts_uuid_identifier_and_identifier_slug(self, mock_fetch, _):
        """Lookup accepts each published article identifier form."""
        mock_fetch.return_value = [
            {
                "id": "UUID-123",
                "identifier": "Article-456",
                "slug": "How-To-Use",
                "title": "How to use",
                "current_published_content_html": "<p>Content</p>",
            }
        ]

        for article_id in ("uuid-123/", "article-456/", "ARTICLE-456-HOW-TO-USE/"):
            with self.subTest(article_id=article_id):
                result = self.module.get_support_article_content.invoke(article_id)
                self.assertIn("ID: UUID-123", result)
                self.assertIn("<p>Content</p>", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_uuid_match_takes_precedence_over_alias_match(self, mock_fetch, _):
        """A UUID match wins when it collides with another article alias."""
        mock_fetch.return_value = [
            {
                "id": "other-id",
                "identifier": "UUID-123",
                "slug": "article",
                "title": "Alias article",
            },
            {"id": "UUID-123", "title": "UUID article"},
        ]

        result = self.module.get_support_article_content.invoke("uuid-123")

        self.assertIn("Title: UUID article", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_not_found_message_describes_accepted_forms(self, mock_fetch, _):
        """Not-found output explains the canonical UUID and aliases."""
        mock_fetch.return_value = [{"id": "UUID-123", "identifier": "article-456"}]

        result = self.module.get_support_article_content.invoke("missing/")

        self.assertEqual(
            result,
            "Article ID missing/ not found. Pass the UUID from the `ID:` field, "
            "the article identifier, or the identifier-slug shown in the article URL.",
        )


if __name__ == "__main__":
    unittest.main()
