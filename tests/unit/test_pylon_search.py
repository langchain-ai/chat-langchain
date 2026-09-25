"""Tests for bounded support article search."""

import json
import unittest
from unittest.mock import patch


class TestSearchSupportArticles(unittest.TestCase):
    """Unit tests for search_support_articles result bounds."""

    @patch("src.tools.pylon_tools._fetch_collections")
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_search_is_bounded_and_compact(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        """Search returns only the requested number of ranked articles."""
        mock_fetch_collections.return_value = {
            "LangSmith Observability": "observability"
        }
        mock_fetch_articles.return_value = [
            {
                "id": f"article-{index}",
                "title": (
                    f"LangSmith observability issue {index} with "
                    "troubleshooting details"
                ),
                "identifier": f"identifier-{index}",
                "slug": f"article-{index}",
                "collection_id": "observability",
                "is_published": True,
                "visibility_config": {"visibility": "public"},
            }
            for index in range(218)
        ]

        from src.tools.pylon_tools import search_support_articles

        serialized = search_support_articles.invoke(
            {"query": "LangSmith observability", "limit": 12}
        )
        result = json.loads(serialized)

        self.assertEqual(result["returned"], 12)
        self.assertEqual(len(result["articles"]), 12)
        self.assertEqual(result["total_matched"], 218)
        self.assertTrue(result["truncated"])
        self.assertLess(len(serialized), 10000)


if __name__ == "__main__":
    unittest.main()
