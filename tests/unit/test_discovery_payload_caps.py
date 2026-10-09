"""Tests for the discovery payload caps on search_support_articles and MCP search results.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import ToolMessage

from src.middleware.tool_result_cap_middleware import ToolResultCapMiddleware


def _article(index, title):
    """Build a Pylon-shaped published article payload."""
    return {
        "id": f"id-{index}",
        "identifier": str(1000 + index),
        "slug": f"slug-{index}",
        "title": title,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "collection_id": "coll-1",
    }


CATALOG = [_article(i, f"Tracing guide {i}") for i in range(40)] + [
    _article(100, "Self hosted upgrade"),
]


class TestSearchSupportArticlesQuery(unittest.TestCase):
    """Unit tests for the required query filter and result cap."""

    def setUp(self):
        import src.tools.pylon_tools as pylon_module

        pylon_module._articles_cache = None
        self.module = pylon_module

    def _invoke(self, **kwargs):
        with patch.object(self.module, "_fetch_all_articles", return_value=CATALOG), patch.object(
            self.module, "_fetch_collections", return_value={"LangSmith Observability": "coll-1"}
        ):
            return json.loads(self.module.search_support_articles.invoke(kwargs))

    def test_results_are_keyword_filtered_and_capped(self):
        """Only title matches are returned, and never more than the hard cap of 25."""
        result = self._invoke(query="tracing")

        self.assertEqual(result["total"], 40)
        self.assertEqual(result["returned"], 25)
        self.assertEqual(len(result["articles"]), 25)
        self.assertTrue(
            all("tracing" in article["title"].lower() for article in result["articles"])
        )

    def test_limit_is_respected(self):
        """An explicit limit below the hard cap bounds the payload further."""
        result = self._invoke(query="tracing", limit=3)

        self.assertEqual(result["returned"], 3)

    def test_no_match_does_not_return_full_catalog(self):
        """A query with no title matches returns an empty result, not the catalog."""
        result = self._invoke(query="kubernetes")

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["articles"], [])
        self.assertEqual(result["query"], "kubernetes")


class TestToolResultCapMiddleware(unittest.TestCase):
    """Unit tests for capping oversized search tool results."""

    def _run(self, tool_name, content):
        middleware = ToolResultCapMiddleware()
        message = ToolMessage(content=content, name=tool_name, tool_call_id="call-1")

        class FakeRequest:
            tool_call = {"name": tool_name, "id": "call-1"}

        async def handler(_request):
            return message

        return asyncio.run(middleware.awrap_tool_call(FakeRequest(), handler))

    def test_extra_results_are_dropped_and_entries_truncated(self):
        """Only the top 10 entries survive and each entry is shortened."""
        entries = [f"Title: page {i}\nContent: {'x' * 5000}" for i in range(20)]
        result = self._run("search_docs_by_lang_chain", "\n\n".join(entries))

        self.assertIn("10 more results omitted", result.content)
        self.assertIn("Title: page 0", result.content)
        self.assertNotIn("Title: page 12", result.content)
        self.assertLess(len(result.content), 30_000)

    def test_uncapped_tools_pass_through(self):
        """Read tools keep their full payload so answers stay grounded."""
        content = "Content: " + "y" * 60_000
        result = self._run("query_docs_filesystem_docs_by_lang_chain", content)

        self.assertEqual(result.content, content)


if __name__ == "__main__":
    unittest.main()
