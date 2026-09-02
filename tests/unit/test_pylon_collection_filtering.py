"""Tests for support article collection filtering."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles

COLLECTIONS = {
    "General": "general-id",
    "OSS (LangChain and LangGraph)": "oss-id",
    "LangSmith Observability": "observability-id",
}

ARTICLES = [
    {
        "id": "article-1",
        "title": "OSS article",
        "identifier": "article-1",
        "slug": "oss-article",
        "collection_id": "oss-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
    {
        "id": "article-2",
        "title": "General article",
        "identifier": "article-2",
        "slug": "general-article",
        "collection_id": "general-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
]


class TestSearchSupportArticleCollectionFiltering(unittest.TestCase):
    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_reversed_parenthetical_order_matches(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "article-1")

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_truncated_parenthetical_matches(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke({"collections": "OSS (LangGraph)"})
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "article-1")

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_invalid_collection_is_skipped_with_warning(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke(
                {"collections": "General, Not A Collection"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "article-2")
        self.assertEqual(
            result["warnings"], ["Unknown collection(s) ignored: Not A Collection"]
        )

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_all_invalid_collections_raise(self, mock_articles, mock_collections):
        with self.assertRaises(ValueError) as context:
            search_support_articles.invoke({"collections": "Unknown, Missing"})

        self.assertIn("No known collection", str(context.exception))


if __name__ == "__main__":
    unittest.main()
