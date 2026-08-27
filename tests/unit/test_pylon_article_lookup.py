"""Tests for support article identifier lookup."""

import importlib
import unittest
from unittest.mock import patch

ARTICLES = [
    {
        "id": "uuid-1",
        "identifier": "1242226068",
        "slug": "how-do-i-configure-checkpointing-in-langgraph",
        "title": "Configure checkpointing in LangGraph",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>Content</p>",
    },
    {
        "id": "uuid-2",
        "identifier": "9065388713",
        "slug": "another-article",
        "title": "Another article",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>Other content</p>",
    },
]


class TestSupportArticleLookup(unittest.TestCase):
    """Unit tests for support article identifier lookup."""

    def setUp(self):
        """Reload the module and reset the article cache before each test."""
        self.module = importlib.import_module("src.tools.pylon_tools")
        self.module._articles_cache = ARTICLES
        self.module._collections_cache = {"Support": "collection-1"}

    def tearDown(self):
        """Reset module caches after each test."""
        self.module._articles_cache = None
        self.module._collections_cache = None

    @patch("src.tools.pylon_tools._fetch_collections", return_value={"Support": "collection-1"})
    def test_lookup_by_uuid(self, mock_collections):
        result = self.module.get_support_article_content.invoke("uuid-1")

        self.assertIn("ID: uuid-1", result)
        self.assertIn("Content", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={"Support": "collection-1"})
    def test_lookup_by_bare_identifier(self, mock_collections):
        result = self.module.get_support_article_content.invoke(" 1242226068 ")

        self.assertIn("ID: uuid-1", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={"Support": "collection-1"})
    def test_lookup_by_identifier_slug(self, mock_collections):
        result = self.module.get_support_article_content.invoke(
            "https://support.langchain.com/articles/1242226068-how-do-i-configure-checkpointing-in-langgraph"
        )

        self.assertIn("ID: uuid-1", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={"Support": "collection-1"})
    def test_unknown_value_suggests_uuid(self, mock_collections):
        result = self.module.get_support_article_content.invoke("unknown-article")

        self.assertIn("uuid-1", result)
        self.assertIn("Retry using the id field", result)


if __name__ == "__main__":
    unittest.main()
