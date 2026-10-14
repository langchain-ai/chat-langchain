"""Tests for cached support article lookup through the listing and read tools."""

import json
import unittest
from unittest.mock import patch

from src.tools import pylon_tools


class TestSupportArticleContent(unittest.TestCase):
    def setUp(self):
        self.article = {
            "id": "12345678-1234-4321-8765-123456789abc",
            "identifier": 42,
            "slug": "support-article",
            "title": "Support article",
            "collection_id": "collection-1",
            "is_published": True,
            "visibility_config": {"visibility": "public"},
            "current_published_content_html": "<p>Article content</p>",
        }
        other_article = {
            **self.article,
            "id": "abcdefab-1234-4321-8765-123456789abc",
            "identifier": 12345678,
            "current_published_content_html": "<p>Other article content</p>",
        }
        self.addCleanup(patch.stopall)
        patch.object(
            pylon_tools, "_articles_cache", [other_article, self.article]
        ).start()
        patch.object(
            pylon_tools, "_collections_cache", {"General": "collection-1"}
        ).start()
        self.mock_get = patch.object(pylon_tools.requests, "get").start()
        listing = json.loads(pylon_tools.search_support_articles.invoke({}))
        self.listed_article = next(
            article
            for article in listing["articles"]
            if article["id"] == self.article["id"]
        )
        self.expected_content = (
            f"ID: {self.article['id']}\n"
            "Title: Support article\n"
            "URL: https://support.langchain.com/articles/42-support-article\n"
            "Collection: General\n\n"
            "Content:\n<p>Article content</p>"
        )

    def tearDown(self):
        self.mock_get.assert_not_called()

    def test_listing_uuid_lookup_takes_precedence_over_numeric_prefix(self):
        result = pylon_tools.get_support_article_content.invoke(
            {"article_id": self.listed_article["id"]}
        )
        self.assertEqual(result, self.expected_content)

    def test_numeric_identifier_lookup(self):
        result = pylon_tools.get_support_article_content.invoke({"article_id": "42"})
        self.assertEqual(result, self.expected_content)

    def test_identifier_slug_lookup(self):
        result = pylon_tools.get_support_article_content.invoke(
            {"article_id": "42-support-article"}
        )
        self.assertEqual(result, self.expected_content)

    def test_listing_url_lookup(self):
        result = pylon_tools.get_support_article_content.invoke(
            {"article_id": self.listed_article["url"]}
        )
        self.assertEqual(result, self.expected_content)

    def test_whitespace_and_url_suffixes(self):
        for article_id in (
            self.listed_article["id"],
            "42",
            "42-support-article",
            self.listed_article["url"],
            self.listed_article["url"] + "/?source=search#content",
        ):
            with self.subTest(article_id=article_id):
                result = pylon_tools.get_support_article_content.invoke(
                    {"article_id": f" \t{article_id}\n "}
                )
                self.assertEqual(result, self.expected_content)

    def test_string_identifier_lookup(self):
        self.article["identifier"] = "42"
        result = pylon_tools.get_support_article_content.invoke({"article_id": "42"})
        self.assertEqual(result, self.expected_content)

    def test_unknown_value_returns_actionable_error(self):
        for article_id in (
            "unknown",
            "999",
            "999-missing-article",
            "https://support.langchain.com/articles/999-missing-article",
            "https://docs.langchain.com/articles/42-support-article",
            "https://support.langchain.com.example.com/articles/42-support-article",
            "Support article",
        ):
            with self.subTest(article_id=article_id):
                result = pylon_tools.get_support_article_content.invoke(
                    {"article_id": article_id}
                )
                self.assertEqual(
                    result,
                    f'Article ID {article_id} not found. Pass the "id" (UUID) field from search_support_articles.',
                )


if __name__ == "__main__":
    unittest.main()
