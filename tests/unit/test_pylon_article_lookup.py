"""Tests for Pylon support article lookup identifiers."""

import json
import unittest
from unittest.mock import patch

ARTICLES = [
    {
        "id": "uuid-1",
        "identifier": 12345,
        "slug": "first-article",
        "title": "First article",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>First content</p>",
    },
    {
        "id": "uuid-2",
        "identifier": "67890",
        "slug": "second-article",
        "title": "Second article",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>Second content</p>",
    },
]


class TestPylonArticleLookup(unittest.TestCase):
    """Unit tests for support article identifier lookup."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_lookup_by_uuid(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": " uuid-1 "})

        self.assertIn("First article", result)
        self.assertIn("First content", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_lookup_by_numeric_identifier(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": "12345"})

        self.assertIn("First article", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_lookup_by_full_support_url(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke(
            {
                "article_id": "https://support.langchain.com/articles/12345-first-article"
            }
        )

        self.assertIn("First article", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_unknown_id_returns_structured_error(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": "unknown"})
        error = json.loads(result)

        self.assertIn("error", error)
        self.assertEqual(error["example_valid_ids"], ["uuid-1", "uuid-2"])


if __name__ == "__main__":
    unittest.main()
