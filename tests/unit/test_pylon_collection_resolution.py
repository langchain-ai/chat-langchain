"""Tests for collection-name resolution in src/tools/pylon_tools.py.

These tests do NOT require network access or LangSmith credentials.
All Pylon API access is mocked via unittest.mock.
"""

import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

import src.tools.pylon_tools as pylon_module

COLLECTION_MAP = {
    "General": "c-general",
    "OSS (LangChain and LangGraph)": "c-oss",
    "LangSmith Observability": "c-obs",
    "LangSmith Evaluation": "c-eval",
    "LangSmith Deployment": "c-deploy",
    "SDKs and APIs": "c-sdk",
    "LangSmith Studio": "c-studio",
    "Self Hosted": "c-self",
    "Troubleshooting": "c-trouble",
    "Security": "c-security",
}


def _article(article_id, collection_id):
    return {
        "id": article_id,
        "title": f"Title {article_id}",
        "identifier": f"id-{article_id}",
        "slug": f"slug-{article_id}",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "collection_id": collection_id,
    }


ARTICLES = [
    _article("a1", "c-oss"),
    _article("a2", "c-deploy"),
    _article("a3", "c-obs"),
]


class TestCollectionResolution(unittest.TestCase):
    """Unit tests for scrambled and multi-collection inputs."""

    def setUp(self):
        patcher_articles = patch.object(
            pylon_module, "_fetch_all_articles", return_value=ARTICLES
        )
        patcher_collections = patch.object(
            pylon_module, "_fetch_collections", return_value=COLLECTION_MAP
        )
        self.addCleanup(patcher_articles.stop)
        self.addCleanup(patcher_collections.stop)
        patcher_articles.start()
        patcher_collections.start()
        # Call the undecorated implementation so args_schema validation does not
        # intercept the deliberately malformed names under test.
        self.search = pylon_module.search_support_articles.func

    # ------------------------------------------------------------------
    # Scrambled collection names
    # ------------------------------------------------------------------

    def test_duplicated_word_scramble_resolves(self):
        """A duplicated-word scramble resolves to the real collection."""
        result = json.loads(self.search("OSS (LangGraph and LangGraph)"))

        self.assertNotIn("error", result)
        self.assertEqual([a["id"] for a in result["articles"]], ["a1"])

    def test_reordered_word_scramble_resolves(self):
        """A word-order scramble resolves to the real collection."""
        result = json.loads(self.search("OSS (LangGraph and LangChain)"))

        self.assertNotIn("error", result)
        self.assertEqual([a["id"] for a in result["articles"]], ["a1"])

    def test_case_insensitive_match_still_works(self):
        """Case-only differences still resolve."""
        result = json.loads(self.search("oss (langchain and langgraph)"))

        self.assertEqual([a["id"] for a in result["articles"]], ["a1"])

    # ------------------------------------------------------------------
    # Genuinely unknown names
    # ------------------------------------------------------------------

    def test_unknown_collection_returns_error_json(self):
        """An unrelated name still returns the not-found error payload."""
        result = json.loads(self.search("Kubernetes Autoscaling"))

        self.assertIn("error", result)
        self.assertIn("Kubernetes Autoscaling", result["error"])
        self.assertIn("Available collections:", result["error"])

    # ------------------------------------------------------------------
    # Comma-joined input
    # ------------------------------------------------------------------

    def test_comma_joined_collections(self):
        """Comma-joined collection names filter to the union of both."""
        result = json.loads(self.search("LangSmith Deployment,LangSmith Observability"))

        self.assertEqual(sorted(a["id"] for a in result["articles"]), ["a2", "a3"])

    def test_comma_joined_collections_with_scramble(self):
        """A scrambled name inside a comma-joined list still resolves."""
        result = json.loads(
            self.search("OSS (LangGraph and LangGraph), LangSmith Deployment")
        )

        self.assertEqual(sorted(a["id"] for a in result["articles"]), ["a1", "a2"])


class TestCollectionArgsSchema(unittest.TestCase):
    """Unit tests for the validated args schema of search_support_articles."""

    def test_schema_advertises_allowed_values(self):
        """The emitted JSON schema enumerates the real collection names."""
        schema = pylon_module.search_support_articles.args_schema.model_json_schema()
        options = schema["properties"]["collections"]["anyOf"]
        enums = [opt["enum"] for opt in options if "enum" in opt]

        self.assertEqual(len(enums), 1)
        self.assertIn("OSS (LangChain and LangGraph)", enums[0])
        self.assertIn("all", enums[0])

    def test_scrambled_value_is_normalized(self):
        """A scrambled name is rewritten to the canonical name during validation."""
        args = pylon_module.SearchSupportArticlesInput(
            collections="OSS (LangGraph and LangGraph)"
        )

        self.assertEqual(args.collections, "OSS (LangChain and LangGraph)")

    def test_comma_joined_value_accepted(self):
        """Comma-joined names pass validation and stay comma-joined."""
        args = pylon_module.SearchSupportArticlesInput(
            collections="LangSmith Deployment, LangSmith Observability"
        )

        self.assertEqual(
            args.collections, "LangSmith Deployment,LangSmith Observability"
        )

    def test_unknown_value_rejected(self):
        """An unknown name fails validation so the caller must retry."""
        with self.assertRaises(ValidationError):
            pylon_module.SearchSupportArticlesInput(
                collections="Kubernetes Autoscaling"
            )


if __name__ == "__main__":
    unittest.main()
