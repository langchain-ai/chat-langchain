"""Tests for support article collection resolution."""

import json
import unittest
from unittest.mock import patch

COLLECTIONS = {
    "General": "collection-general",
    "OSS (LangChain and LangGraph)": "collection-oss",
}

ARTICLES = [
    {
        "id": "article-1",
        "title": "Getting started",
        "identifier": "identifier-1",
        "slug": "getting-started",
        "collection_id": "collection-oss",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    }
]


class TestSearchSupportArticlesCollections(unittest.TestCase):
    """Unit tests for support article collection resolution."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_transposed_parenthesized_name_resolves(
        self, mock_articles, mock_collections
    ):
        from src.tools.pylon_tools import search_support_articles

        result = json.loads(
            search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "article-1")

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_valid_and_invalid_names_return_articles_and_warning(
        self, mock_articles, mock_collections
    ):
        from src.tools.pylon_tools import search_support_articles

        result = json.loads(
            search_support_articles.invoke(
                {"collections": "General, OSS (LangChain and LangGraph), Junk"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["unrecognized_collections"], ["Junk"])
        self.assertEqual(result["available_collections"], sorted(COLLECTIONS))

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_all_invalid_names_return_error(self, mock_articles, mock_collections):
        from src.tools.pylon_tools import search_support_articles

        result = json.loads(
            search_support_articles.invoke({"collections": "Junk, Missing"})
        )

        self.assertIn("error", result)
        self.assertIn("Junk", result["error"])
        self.assertNotIn("unrecognized_collections", result)


if __name__ == "__main__":
    unittest.main()
