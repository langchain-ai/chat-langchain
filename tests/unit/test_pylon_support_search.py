"""Tests for support article search."""

import json
import unittest
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles

COLLECTIONS = {"General": "general", "Troubleshooting": "troubleshooting"}


def _article(article_id, title, collection_id="general"):
    return {
        "id": article_id,
        "title": title,
        "identifier": article_id,
        "slug": title.lower().replace(" ", "-"),
        "collection_id": collection_id,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    }


class SupportArticleSearchTests(unittest.TestCase):
    def call_search(self, query, articles, collections="all"):
        with (
            patch("src.tools.pylon_tools._fetch_all_articles", return_value=articles),
            patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS),
        ):
            return json.loads(search_support_articles.func(query, collections))

    def test_matches_titles_by_query_tokens(self):
        result = self.call_search(
            "API key",
            [
                _article("1", "Rotate API keys"),
                _article("2", "API authentication"),
                _article("3", "Deployment settings"),
            ],
        )

        self.assertEqual(result["total"], 2)
        self.assertEqual([article["id"] for article in result["articles"]], ["1", "2"])

    def test_returns_empty_result_when_no_title_matches(self):
        result = self.call_search("billing", [_article("1", "Deployment settings")])

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["articles"], [])
        self.assertIn("No titles matched", result["note"])

    def test_caps_results_but_preserves_uncapped_total(self):
        articles = [
            _article(str(index), f"Deployment guide {index}") for index in range(25)
        ]

        result = self.call_search("deployment", articles)

        self.assertEqual(result["total"], 25)
        self.assertEqual(len(result["articles"]), 20)
        self.assertIn("capped", result["note"])

    def test_preserves_response_and_article_key_sets(self):
        result = self.call_search("deployment", [_article("1", "Deployment settings")])

        self.assertEqual(set(result), {"collections", "total", "articles", "note"})
        self.assertEqual(
            set(result["articles"][0]),
            {"id", "title", "url", "collection"},
        )
        self.assertNotIn("collection_id", result["articles"][0])


if __name__ == "__main__":
    unittest.main()
