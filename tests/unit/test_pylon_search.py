"""Tests for support article collection resolution."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles

COLLECTIONS = {
    "General": "general-id",
    "OSS (LangChain and LangGraph)": "oss-id",
    "LangSmith Deployment": "deployment-id",
    "LangSmith Observability": "observability-id",
}


def _article(collection_id: str) -> dict:
    return {
        "id": f"article-{collection_id}",
        "title": "An article",
        "identifier": "identifier",
        "slug": "article",
        "collection_id": collection_id,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    }


class TestSearchSupportArticlesCollectionResolution(unittest.TestCase):
    def _search(self, collections: str) -> dict:
        with (
            patch(
                "src.tools.pylon_tools._fetch_all_articles",
                return_value=[_article("oss-id"), _article("deployment-id")],
            ),
            patch(
                "src.tools.pylon_tools._fetch_collections",
                return_value=COLLECTIONS,
            ),
        ):
            return json.loads(search_support_articles.func(collections))

    def test_word_order_flipped_name_resolves(self):
        result = self._search("OSS (LangGraph and LangChain)")

        self.assertEqual(result["total"], 1)
        self.assertEqual(
            result["articles"][0]["collection"], "OSS (LangChain and LangGraph)"
        )
        self.assertNotIn("unresolved_collections", result)

    def test_truncated_name_resolves(self):
        result = self._search("OSS (LangGraph)")

        self.assertEqual(result["total"], 1)
        self.assertEqual(
            result["articles"][0]["collection"], "OSS (LangChain and LangGraph)"
        )

    def test_invalid_name_preserves_valid_results(self):
        result = self._search("LangSmith Deployment, Billing")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "LangSmith Deployment")
        self.assertEqual(result["unresolved_collections"], ["Billing"])

    def test_all_invalid_names_return_error(self):
        result = self._search("Billing")

        self.assertIn("error", result)
        self.assertIn("General", result["error"])
        self.assertNotIn("articles", result)


if __name__ == "__main__":
    unittest.main()
