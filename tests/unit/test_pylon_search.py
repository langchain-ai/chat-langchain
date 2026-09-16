"""Tests for tolerant Pylon collection filtering."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles

COLLECTIONS = {
    "General": "general-id",
    "OSS (LangChain and LangGraph)": "oss-id",
    "Troubleshooting": "troubleshooting-id",
}


def _article(article_id: str, collection_id: str) -> dict:
    return {
        "id": article_id,
        "title": f"Article {article_id}",
        "identifier": article_id,
        "slug": f"article-{article_id}",
        "collection_id": collection_id,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    }


class TestSearchSupportArticles(unittest.TestCase):
    """Unit tests for search_support_articles collection filtering."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article("oss-article", "oss-id")],
    )
    def test_transposed_oss_collection_name_resolves(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "OSS (LangChain and LangGraph)")

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article("troubleshooting-article", "troubleshooting-id")],
    )
    def test_mixed_known_and_unknown_collections(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke({"collections": "Billing,Troubleshooting"})
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["unmatched_collections"], ["Billing"])

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article("general-article", "general-id")],
    )
    def test_all_unknown_collections_return_error(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke({"collections": "Billing"})
        )

        self.assertIn("error", result)
        self.assertIn("Billing", result["error"])


if __name__ == "__main__":
    unittest.main()
