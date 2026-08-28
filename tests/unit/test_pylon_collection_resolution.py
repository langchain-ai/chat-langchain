"""Tests for support article collection name resolution."""

import json
import unittest
from unittest.mock import patch

COLLECTIONS = {
    "General": "general-id",
    "OSS (LangChain and LangGraph)": "oss-id",
}


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


class TestPylonCollectionResolution(unittest.TestCase):
    """Unit tests for tolerant collection resolution."""

    def _search(self, collections):
        import src.tools.pylon_tools as pylon_module

        with patch.object(
            pylon_module,
            "_fetch_all_articles",
            return_value=[_article("a1", "oss-id"), _article("a2", "general-id")],
        ), patch.object(pylon_module, "_fetch_collections", return_value=COLLECTIONS):
            return json.loads(
                pylon_module.search_support_articles.invoke({"collections": collections})
            )

    def test_word_order_paraphrase_resolves(self):
        result = self._search("OSS (LangGraph and LangChain)")

        self.assertEqual(result["resolved_collections"], ["OSS (LangChain and LangGraph)"])

    def test_duplicated_word_paraphrase_resolves(self):
        result = self._search("OSS (LangGraph and LangGraph)")

        self.assertEqual(result["resolved_collections"], ["OSS (LangChain and LangGraph)"])

    def test_general_and_oss_resolve_together(self):
        result = self._search("OSS (LangGraph),General")

        self.assertEqual(
            result["resolved_collections"],
            ["OSS (LangChain and LangGraph)", "General"],
        )
        self.assertEqual(result["total"], 2)

    def test_unknown_name_does_not_discard_valid_collection(self):
        result = self._search("Unknown,General")

        self.assertEqual(result["resolved_collections"], ["General"])
        self.assertEqual(result["unresolved_collections"], ["Unknown"])
        self.assertEqual(result["total"], 1)

    def test_all_unknown_names_return_error(self):
        result = self._search("Unknown,Also Unknown")

        self.assertIn("error", result)
        self.assertNotIn("resolved_collections", result)


if __name__ == "__main__":
    unittest.main()
