"""Tests for tolerant collection resolution in search_support_articles."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles

COLLECTION_MAP = {
    "General": "general-id",
    "OSS (LangChain and LangGraph)": "oss-id",
}

ARTICLES = [
    {
        "id": "oss-article",
        "title": "OSS article",
        "identifier": "oss-article",
        "slug": "oss-article",
        "collection_id": "oss-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
    {
        "id": "general-article",
        "title": "General article",
        "identifier": "general-article",
        "slug": "general-article",
        "collection_id": "general-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
]


class TestPylonCollectionResolution(unittest.TestCase):
    """Test collection name matching and partial resolution."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTION_MAP)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_paraphrased_oss_names_resolve(self, mock_articles, mock_collections):
        """Reordered, duplicated, and case-folded names resolve to OSS."""
        for requested_name in (
            "OSS (LangGraph and LangChain)",
            "OSS (LangGraph and LangGraph)",
            "oss (langchain and langgraph)",
        ):
            with self.subTest(requested_name=requested_name):
                result = json.loads(search_support_articles.func(requested_name))
                self.assertEqual(result["total"], 1)
                self.assertEqual(result["articles"][0]["collection"], "OSS (LangChain and LangGraph)")

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTION_MAP)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_partial_collection_resolution_returns_results(self, mock_articles, mock_collections):
        """A valid collection still returns results when another name is unresolved."""
        result = json.loads(search_support_articles.func("OSS (LangGraph),General"))

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "General")
        self.assertEqual(result["unresolved_collections"], ["OSS (LangGraph)"])
        self.assertEqual(result["available_collections"], list(COLLECTION_MAP))


if __name__ == "__main__":
    unittest.main()
