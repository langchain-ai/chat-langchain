"""Tests for lenient collection resolution in search_support_articles.

These tests do NOT require network access or LangSmith credentials.
The Pylon fetch helpers are mocked via unittest.mock.
"""

import json
import unittest
from unittest.mock import patch

import src.tools.pylon_tools as pylon_module

COLLECTION_MAP = {
    "General": "c-general",
    "OSS (LangChain and LangGraph)": "c-oss",
    "LangSmith Deployment": "c-deploy",
    "Security": "c-security",
}


def _article(article_id, title, collection_id):
    return {
        "id": article_id,
        "title": title,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "identifier": article_id,
        "slug": title.lower().replace(" ", "-"),
        "collection_id": collection_id,
    }


ARTICLES = [
    _article("a1", "Using LangGraph", "c-oss"),
    _article("a2", "Deploying a graph", "c-deploy"),
    _article("a3", "Managing your org", "c-general"),
]


class TestCollectionResolution(unittest.TestCase):
    """Unit tests for how requested collection labels are resolved."""

    def _search(self, collections):
        with patch.object(pylon_module, "_fetch_all_articles", return_value=ARTICLES), \
                patch.object(pylon_module, "_fetch_collections", return_value=COLLECTION_MAP):
            return json.loads(
                pylon_module.search_support_articles.invoke(
                    {"collections": collections}
                )
            )

    def test_reversed_parenthesised_label_resolves(self):
        """A reversed OSS label still resolves to the real collection."""
        result = self._search("OSS (LangGraph and LangChain)")

        self.assertEqual(result["unresolved_collections"], [])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "a1")

    def test_unknown_label_skipped_but_valid_one_returns(self):
        """A nonexistent label is skipped instead of aborting the whole call."""
        result = self._search("Billing,LangSmith Deployment")

        self.assertEqual(result["unresolved_collections"], ["Billing"])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["id"], "a2")
        self.assertIn("Billing", result["note"])

    def test_all_unresolved_returns_error(self):
        """When nothing resolves, an error payload with the available labels is returned."""
        result = self._search("Billing,Nonsense Collection")

        self.assertIn("error", result)
        self.assertIn("Billing", result["error"])
        self.assertIn("OSS (LangChain and LangGraph)", result["error"])


class TestNormalizeCollectionName(unittest.TestCase):
    """Unit tests for _normalize_collection_name()."""

    def test_parenthesised_tokens_are_order_insensitive(self):
        self.assertEqual(
            pylon_module._normalize_collection_name("OSS (LangGraph and LangChain)"),
            pylon_module._normalize_collection_name("OSS (LangChain and LangGraph)"),
        )

    def test_whitespace_and_case_collapsed(self):
        self.assertEqual(
            pylon_module._normalize_collection_name("  langsmith   DEPLOYMENT "),
            pylon_module._normalize_collection_name("LangSmith Deployment"),
        )


if __name__ == "__main__":
    unittest.main()
