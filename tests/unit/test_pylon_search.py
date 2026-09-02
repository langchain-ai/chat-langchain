"""Tests for support article collection resolution."""

import json
import unittest
from unittest.mock import patch

from langchain_core.tools import ToolException

from src.tools import pylon_tools

ARTICLES = [
    {
        "id": "article-1",
        "title": "LangChain article",
        "identifier": "article-1",
        "slug": "langchain-article",
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

COLLECTIONS = {
    "OSS (LangChain and LangGraph)": "oss-id",
    "General": "general-id",
}


class TestSearchSupportArticlesCollections(unittest.TestCase):
    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_transposed_name_matches_collection(self, mock_articles, mock_collections):
        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "OSS (LangChain and LangGraph)")

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_partial_match_returns_results_and_warning(self, mock_articles, mock_collections):
        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "General, Billing"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["warnings"], ["Collection 'Billing' not found"])
        self.assertEqual(result["available_collections"], ["General", "OSS (LangChain and LangGraph)"])

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_all_invalid_collections_raise_tool_exception(self, mock_articles, mock_collections):
        with self.assertRaises(ToolException) as context:
            pylon_tools.search_support_articles.invoke({"collections": "Billing"})

        self.assertIn("Billing", str(context.exception))
        self.assertIn("General", str(context.exception))


if __name__ == "__main__":
    unittest.main()
