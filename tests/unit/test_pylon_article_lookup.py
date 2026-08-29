"""Tests for support article identifier resolution."""

import json
import unittest
from unittest.mock import patch

import src.tools.pylon_tools as pylon_tools

ARTICLES = [
    {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "identifier": "1234567890",
        "slug": "example-article",
        "title": "Example article",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>Content</p>",
    }
]


class TestSupportArticleLookup(unittest.TestCase):
    def setUp(self):
        pylon_tools._last_search_articles = ARTICLES

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_accepts_uuid_numeric_identifier_and_url(
        self, mock_fetch, mock_collections
    ):
        expected = "Example article"
        for article_id in (
            ARTICLES[0]["id"],
            ARTICLES[0]["identifier"],
            "https://support.langchain.com/articles/1234567890-example-article",
        ):
            result = pylon_tools.get_support_article_content.invoke(
                {"article_id": article_id}
            )
            self.assertIn(expected, result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_unknown_identifier_returns_valid_uuid_ids(
        self, mock_fetch, mock_collections
    ):
        result = pylon_tools.get_support_article_content.invoke(
            {"article_id": "9999999999"}
        )
        payload = json.loads(result)
        self.assertEqual(payload["valid_article_ids"], [ARTICLES[0]["id"]])
        self.assertIn("valid UUID", payload["message"])


if __name__ == "__main__":
    unittest.main()
