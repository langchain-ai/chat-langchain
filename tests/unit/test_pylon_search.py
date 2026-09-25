"""Tests for support article search filtering and result limits."""

import json
import unittest
from unittest.mock import patch

from src.tools import pylon_tools


def _article(index: int) -> dict:
    return {
        "id": f"article-{index}",
        "title": f"Deployment error troubleshooting {index}",
        "identifier": f"identifier-{index}",
        "slug": f"deployment-error-{index}",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "collection_id": "deployment",
    }


class TestSearchSupportArticles(unittest.TestCase):
    """Unit tests for search_support_articles result filtering."""

    @patch.object(
        pylon_tools,
        "_fetch_collections",
        return_value={"LangSmith Deployment": "deployment"},
    )
    @patch.object(pylon_tools, "_fetch_all_articles")
    def test_caps_matching_articles(self, mock_fetch_articles, mock_fetch_collections):
        mock_fetch_articles.return_value = [_article(index) for index in range(20)]

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"query": "deployment error", "collections": "all"}
            )
        )

        self.assertEqual(result["matched"], 20)
        self.assertEqual(result["kept"], pylon_tools.MAX_SEARCH_RESULTS)
        self.assertEqual(len(result["articles"]), pylon_tools.MAX_SEARCH_RESULTS)
        self.assertTrue(
            all(
                set(article) == {"id", "title", "url"} for article in result["articles"]
            )
        )

    @patch.object(
        pylon_tools,
        "_fetch_collections",
        return_value={"LangSmith Deployment": "deployment"},
    )
    @patch.object(pylon_tools, "_fetch_all_articles")
    def test_no_title_matches_returns_empty_articles(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        mock_fetch_articles.return_value = [_article(index) for index in range(20)]

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"query": "billing", "collections": "all"}
            )
        )

        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["kept"], 0)
        self.assertEqual(result["articles"], [])


if __name__ == "__main__":
    unittest.main()
