"""Tests for collection resolution in src/tools/pylon_tools.py."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles

COLLECTION_MAP = {
    "OSS (LangChain and LangGraph)": "oss-id",
    "Troubleshooting": "troubleshooting-id",
}

ARTICLES = [
    {
        "id": "article-1",
        "title": "Troubleshooting article",
        "identifier": "troubleshooting-article",
        "slug": "troubleshooting-article",
        "collection_id": "troubleshooting-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
    {
        "id": "article-2",
        "title": "OSS article",
        "identifier": "oss-article",
        "slug": "oss-article",
        "collection_id": "oss-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
]


class TestPylonCollectionResolution(unittest.TestCase):
    """Test tolerant collection resolution for support article search."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTION_MAP)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_transposed_collection_name_resolves(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.func("OSS (LangGraph and LangChain)")
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(
            result["articles"][0]["collection"], "OSS (LangChain and LangGraph)"
        )

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTION_MAP)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_mixed_collections_returns_matches_and_unmatched(
        self, mock_articles, mock_collections
    ):
        result = json.loads(search_support_articles.func("Billing,Troubleshooting"))

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "Troubleshooting")
        self.assertEqual(result["unmatched_collections"], ["Billing"])
        self.assertEqual(result["available_collections"], sorted(COLLECTION_MAP))

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTION_MAP)
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_all_unknown_collections_returns_actionable_error(
        self, mock_articles, mock_collections
    ):
        result = json.loads(search_support_articles.func("Billing,Payments"))

        self.assertIn(
            "None of the requested collection names resolved", result["error"]
        )
        self.assertEqual(result["available_collections"], sorted(COLLECTION_MAP))
        self.assertIn("Retry", result["instruction"])


if __name__ == "__main__":
    unittest.main()
