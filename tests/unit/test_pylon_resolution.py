"""Tests for tolerant Pylon article and collection resolution."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import get_support_article_content, search_support_articles

ARTICLE = {
    "id": "uuid-123",
    "identifier": 1242226068,
    "slug": "getting-started",
    "title": "Getting Started",
    "collection_id": "collection-1",
    "is_published": True,
    "visibility_config": {"visibility": "public"},
    "current_published_content_html": "<p>Welcome</p>",
}


class TestPylonResolution(unittest.TestCase):
    @patch("src.tools.pylon_tools._fetch_collections", return_value={"General": "collection-1"})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_article_lookup_accepts_numeric_identifier(self, mock_articles, mock_collections):
        result = get_support_article_content.invoke({"article_id": "1242226068"})

        self.assertIn("Title: Getting Started", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={"General": "collection-1"})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_article_lookup_accepts_identifier_slug(self, mock_articles, mock_collections):
        result = get_support_article_content.invoke(
            {"article_id": "1242226068-getting-started"}
        )

        self.assertIn("Title: Getting Started", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={"General": "collection-1"})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_article_lookup_accepts_uuid(self, mock_articles, mock_collections):
        result = get_support_article_content.invoke({"article_id": "UUID-123"})

        self.assertIn("Title: Getting Started", result)

    @patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={"OSS (LangChain and LangGraph)": "collection-1"},
    )
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE])
    def test_collection_lookup_accepts_swapped_word_order(
        self, mock_articles, mock_collections
    ):
        result = json.loads(
            search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "OSS (LangChain and LangGraph)")
        self.assertNotIn("unresolved_collections", result)


if __name__ == "__main__":
    unittest.main()
