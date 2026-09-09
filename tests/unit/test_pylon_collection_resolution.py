"""Tests for support article collection resolution."""

import json
import unittest
from unittest.mock import patch

COLLECTION_MAP = {
    "General": "collection-general",
    "OSS (LangChain and LangGraph)": "collection-oss",
}

ARTICLES = [
    {
        "id": "article-general",
        "title": "General article",
        "identifier": "general-article",
        "slug": "general-article",
        "collection_id": "collection-general",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
    {
        "id": "article-oss",
        "title": "OSS article",
        "identifier": "oss-article",
        "slug": "oss-article",
        "collection_id": "collection-oss",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
]


class TestSupportArticleCollectionResolution(unittest.TestCase):
    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTION_MAP)
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        side_effect=lambda: [article.copy() for article in ARTICLES],
    )
    def test_unresolved_collection_does_not_discard_resolved_articles(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import search_support_articles

        result = json.loads(
            search_support_articles.invoke({"collections": "Billing,General"})
        )

        self.assertEqual(result["unresolved_collections"], ["Billing"])
        self.assertEqual(
            [article["id"] for article in result["articles"]], ["article-general"]
        )

    @patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTION_MAP)
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        side_effect=lambda: [article.copy() for article in ARTICLES],
    )
    def test_normalized_oss_collection_name_resolves(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import search_support_articles

        result = json.loads(
            search_support_articles.invoke({"collections": "OSS (LangGraph)"})
        )

        self.assertEqual(
            result["unresolved_collections"]
            if "unresolved_collections" in result
            else [],
            [],
        )
        self.assertEqual(
            [article["id"] for article in result["articles"]], ["article-oss"]
        )


if __name__ == "__main__":
    unittest.main()
