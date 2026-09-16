"""Tests for get_support_article_content lookup behavior.

Verifies that article lookup accepts both UUID `id` and URL-slug
identifiers, and that the not-found path returns the recovery-hint message.
"""

import unittest
from unittest.mock import patch

SAMPLE_ARTICLE = {
    "id": "uuid-1234",
    "identifier": "7908986507",
    "slug": "attaching-per-message-context-metadata",
    "title": "Attaching per-message context metadata",
    "collection_id": "coll-1",
    "current_published_content_html": "<p>body</p>",
}


class TestGetSupportArticleContentLookup(unittest.TestCase):
    """Unit tests for get_support_article_content article matching."""

    def setUp(self):
        import src.tools.pylon_tools as pylon_module
        pylon_module._articles_cache = None
        pylon_module._collections_cache = None
        self.module = pylon_module

    def _invoke(self, article_id):
        # Tools created via @tool expose the underlying function as .func
        return self.module.get_support_article_content.func(article_id)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[SAMPLE_ARTICLE])
    def test_lookup_by_uuid(self, _articles, _collections):
        result = self._invoke("uuid-1234")
        self.assertIn("ID: uuid-1234", result)
        self.assertIn("Title: Attaching per-message context metadata", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[SAMPLE_ARTICLE])
    def test_lookup_by_identifier_slug(self, _articles, _collections):
        result = self._invoke("7908986507-attaching-per-message-context-metadata")
        self.assertIn("ID: uuid-1234", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[SAMPLE_ARTICLE])
    def test_lookup_by_bare_identifier(self, _articles, _collections):
        result = self._invoke("7908986507")
        self.assertIn("ID: uuid-1234", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[SAMPLE_ARTICLE])
    def test_not_found_returns_recovery_hint(self, _articles, _collections):
        result = self._invoke("does-not-exist")
        self.assertIn("not found", result)
        self.assertIn("'id' field (UUID)", result)
        self.assertIn("not the URL slug", result)


if __name__ == "__main__":
    unittest.main()
