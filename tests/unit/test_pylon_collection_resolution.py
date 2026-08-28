"""Tests for Pylon support article collection resolution."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles


def _article(article_id, collection_id):
    return {
        "id": article_id,
        "title": f"Article {article_id}",
        "identifier": article_id,
        "slug": f"article-{article_id}",
        "collection_id": collection_id,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    }


class TestSearchSupportArticlesCollectionResolution(unittest.TestCase):
    """Unit tests for collection matching and warnings."""

    @patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={
            "OSS (LangChain and LangGraph)": "oss-id",
            "General": "general-id",
        },
    )
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article("oss-article", "oss-id")],
    )
    def test_swapped_order_oss_name_resolves(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(
            result["articles"][0]["collection"], "OSS (LangChain and LangGraph)"
        )
        self.assertNotIn("warnings", result)

    @patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={
            "OSS (LangChain and LangGraph)": "oss-id",
            "General": "general-id",
        },
    )
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article("general-article", "general-id")],
    )
    def test_unresolved_sibling_is_ignored(self, mock_articles, mock_collections):
        result = json.loads(
            search_support_articles.invoke({"collections": "OSS (LangGraph),General"})
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "General")
        self.assertEqual(
            result["warnings"], ["Collection 'OSS (LangGraph)' not found; ignored."]
        )

    @patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={"General": "general-id"},
    )
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article("general-article", "general-id")],
    )
    def test_wholly_unknown_name_returns_not_found(
        self, mock_articles, mock_collections
    ):
        result = json.loads(search_support_articles.invoke({"collections": "Unknown"}))

        self.assertIn("error", result)
        self.assertIn("Collection 'Unknown' not found.", result["error"])


if __name__ == "__main__":
    unittest.main()
