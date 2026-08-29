"""Tests for support article collection resolution."""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

COLLECTIONS = {
    "LangSmith Deployment": "deployment-id",
    "Troubleshooting": "troubleshooting-id",
    "OSS (LangChain and LangGraph)": "oss-id",
}

ARTICLES = [
    {
        "id": "deployment-article",
        "title": "Deployment article",
        "identifier": "deployment",
        "slug": "article",
        "collection_id": "deployment-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
    {
        "id": "troubleshooting-article",
        "title": "Troubleshooting article",
        "identifier": "troubleshooting",
        "slug": "article",
        "collection_id": "troubleshooting-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
    {
        "id": "oss-article",
        "title": "OSS article",
        "identifier": "oss",
        "slug": "article",
        "collection_id": "oss-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
]


class TestSearchSupportArticleCollectionResolution(unittest.TestCase):
    """Unit tests for support article collection resolution."""

    def setUp(self):
        """Reset article cache before each test."""
        import src.tools.pylon_tools as pylon_module

        pylon_module._articles_cache = None
        pylon_module._collections_cache = None
        self.module = pylon_module

    def search(self, collections):
        """Search mocked support articles."""
        with (
            patch.object(
                self.module, "_fetch_all_articles", return_value=deepcopy(ARTICLES)
            ),
            patch.object(self.module, "_fetch_collections", return_value=COLLECTIONS),
        ):
            return json.loads(
                self.module.search_support_articles.invoke({"collections": collections})
            )

    def test_all_in_list_searches_all_collections(self):
        """The all token overrides other collection tokens."""
        result = self.search("LangSmith Deployment,Troubleshooting,all")

        self.assertEqual(result["total"], 3)
        self.assertCountEqual(
            [article["id"] for article in result["articles"]],
            ["deployment-article", "troubleshooting-article", "oss-article"],
        )

    def test_oss_word_order_variant_resolves(self):
        """The OSS collection resolves despite swapped product names."""
        result = self.search("OSS (LangGraph and LangChain)")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "oss-article")
        self.assertEqual(result["unresolved_collections"], [])

    def test_unknown_collection_returns_error(self):
        """A wholly unknown collection retains the error envelope."""
        result = self.search("Unknown collection")

        self.assertIn("error", result)
        self.assertIn("Unknown collection", result["error"])

    def test_mixed_valid_and_unknown_collections_returns_partial_results(self):
        """Valid collections return results while unknown names are reported."""
        result = self.search("Troubleshooting,Unknown collection")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "troubleshooting-article")
        self.assertEqual(result["unresolved_collections"], ["Unknown collection"])
        self.assertEqual(result["available_collections"], sorted(COLLECTIONS))


if __name__ == "__main__":
    unittest.main()
