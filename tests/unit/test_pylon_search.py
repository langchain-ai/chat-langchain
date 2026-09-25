"""Tests for support article collection resolution."""

import json
import unittest
from unittest.mock import patch

from src.tools import pylon_tools


class TestSearchSupportArticles(unittest.TestCase):
    """Unit tests for search_support_articles collection filtering."""

    def setUp(self):
        pylon_tools._articles_cache = None
        pylon_tools._collections_cache = None

    @patch("src.tools.pylon_tools._fetch_collections")
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_partial_collection_match_reports_unrecognized_name(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        mock_fetch_collections.return_value = {
            "General": "general-id",
            "OSS (LangChain and LangGraph)": "oss-id",
        }
        mock_fetch_articles.return_value = [
            {
                "id": "article-id",
                "title": "Article",
                "identifier": "identifier",
                "slug": "article",
                "collection_id": "general-id",
                "is_published": True,
                "visibility_config": {"visibility": "public"},
            }
        ]

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "General,Invented Collection"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["unrecognized_collections"], ["Invented Collection"])

    @patch("src.tools.pylon_tools._fetch_collections")
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_token_order_match_and_zero_articles_include_unrecognized(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        mock_fetch_collections.return_value = {
            "OSS (LangChain and LangGraph)": "oss-id",
        }
        mock_fetch_articles.return_value = [
            {
                "id": "article-id",
                "title": "Article",
                "identifier": "identifier",
                "slug": "article",
                "collection_id": "other-id",
                "is_published": True,
                "visibility_config": {"visibility": "public"},
            }
        ]

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {
                    "collections": (
                        "OSS (LangGraph and LangChain),Unknown Collection"
                    )
                }
            )
        )

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["articles"], [])
        self.assertEqual(result["unrecognized_collections"], ["Unknown Collection"])

    @patch("src.tools.pylon_tools._fetch_collections")
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_all_unrecognized_collections_return_error(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        mock_fetch_collections.return_value = {"General": "general-id"}
        mock_fetch_articles.return_value = [
            {
                "id": "article-id",
                "title": "Article",
                "identifier": "identifier",
                "slug": "article",
                "collection_id": "general-id",
                "is_published": True,
                "visibility_config": {"visibility": "public"},
            }
        ]

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "Unknown Collection"}
            )
        )

        self.assertIn("error", result)
        self.assertIn("General", result["error"])

    @patch("src.tools.pylon_tools._fetch_collections")
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[])
    def test_empty_api_response_reports_unrecognized_name(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        mock_fetch_collections.return_value = {"General": "general-id"}

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "General,Unknown Collection"}
            )
        )

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["unrecognized_collections"], ["Unknown Collection"])


if __name__ == "__main__":
    unittest.main()
