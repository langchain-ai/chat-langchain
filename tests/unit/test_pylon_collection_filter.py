"""Tests for support article collection filtering."""

import json
import unittest
from unittest.mock import patch

from src.tools import pylon_tools


class TestPylonCollectionFilter(unittest.TestCase):
    """Unit tests for collection filter resolution."""

    def setUp(self):
        pylon_tools._articles_cache = None
        pylon_tools._collections_cache = None
        self.collections = {
            "General": "general-id",
            "OSS (LangChain and LangGraph)": "oss-id",
            "Troubleshooting": "troubleshooting-id",
            "LangSmith Deployment": "deployment-id",
        }
        self.articles = [
            self._article("general", "General", "general-id"),
            self._article("oss", "OSS", "oss-id"),
            self._article("troubleshooting", "Troubleshooting", "troubleshooting-id"),
            self._article("deployment", "Deployment", "deployment-id"),
        ]

    @staticmethod
    def _article(identifier, title, collection_id):
        return {
            "id": identifier,
            "identifier": identifier,
            "slug": title.lower(),
            "title": title,
            "collection_id": collection_id,
            "is_published": True,
            "visibility_config": {"visibility": "public"},
        }

    def _search(self, collections):
        with (
            patch.object(
                pylon_tools, "_fetch_all_articles", return_value=self.articles
            ),
            patch.object(
                pylon_tools, "_fetch_collections", return_value=self.collections
            ),
        ):
            return json.loads(
                pylon_tools.search_support_articles.invoke({"collections": collections})
            )

    def test_all_in_comma_list_returns_unfiltered_results(self):
        result = self._search("LangSmith Deployment,Troubleshooting,all")

        self.assertEqual(result["total"], 4)
        self.assertNotIn("warning", result)

    def test_oss_paraphrase_resolves_to_live_collection(self):
        result = self._search("OSS (LangGraph and LangChain),Troubleshooting")

        self.assertEqual(result["total"], 2)
        self.assertEqual(
            {article["collection"] for article in result["articles"]},
            {"OSS (LangChain and LangGraph)", "Troubleshooting"},
        )

    def test_unresolved_collection_returns_partial_results_with_warning(self):
        result = self._search("General,TotallyMadeUp")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["collection"], "General")
        self.assertIn("TotallyMadeUp", result["warning"])
        self.assertIn("General", result["warning"])


if __name__ == "__main__":
    unittest.main()
