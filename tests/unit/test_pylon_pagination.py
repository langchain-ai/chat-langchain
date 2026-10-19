"""Tests for _fetch_all_articles() pagination in src/tools/pylon_tools.py.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import requests
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.middleware.tool_retry_middleware import ToolRetryMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(data, next_cursor=None):
    """Build a mock requests.Response whose .json() returns a Pylon-shaped body."""
    body = {"data": data}
    if next_cursor is not None:
        body["next"] = next_cursor
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _make_meta_response(data, next_cursor=None):
    """Build a mock response that uses the meta.next pagination shape."""
    body = {"data": data, "meta": {}}
    if next_cursor is not None:
        body["meta"]["next"] = next_cursor
    mock_resp = MagicMock()
    mock_resp.json.return_value = body
    mock_resp.raise_for_status.return_value = None
    return mock_resp


ARTICLE_PAGE_1 = [
    {"id": "a1", "title": "Article 1"},
    {"id": "a2", "title": "Article 2"},
]
ARTICLE_PAGE_2 = [
    {"id": "a3", "title": "Article 3"},
    {"id": "a4", "title": "Article 4"},
]
ARTICLE_PAGE_3 = [{"id": "a5", "title": "Article 5"}]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFetchAllArticlesPagination(unittest.TestCase):
    """Unit tests for _fetch_all_articles() pagination behaviour."""

    def setUp(self):
        """Reload the module and reset the global cache before each test."""
        # Ensure a clean module state so _articles_cache starts as None
        import src.tools.pylon_tools as pylon_module

        pylon_module._articles_cache = None
        self.module = pylon_module

    # ------------------------------------------------------------------
    # Single-page case (no "next" key -> stops after one request)
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_single_page_no_next(self, mock_get, mock_kb_id, mock_api_key):
        """When no 'next' cursor is present, exactly one HTTP request is made."""
        mock_get.return_value = _make_response(ARTICLE_PAGE_1, next_cursor=None)

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1)
        mock_get.assert_called_once()

    # ------------------------------------------------------------------
    # Multi-page case (two pages)
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_two_pages_collected(self, mock_get, mock_kb_id, mock_api_key):
        """Articles from both pages are combined into a single list."""
        mock_get.side_effect = [
            _make_response(ARTICLE_PAGE_1, next_cursor="cursor-abc"),
            _make_response(ARTICLE_PAGE_2, next_cursor=None),
        ]

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1 + ARTICLE_PAGE_2)
        self.assertEqual(mock_get.call_count, 2)

        # Second call must pass cursor as query parameter
        _, kwargs = mock_get.call_args_list[1]
        self.assertEqual(kwargs.get("params", {}).get("cursor"), "cursor-abc")

    # ------------------------------------------------------------------
    # Three-page case
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_three_pages_collected(self, mock_get, mock_kb_id, mock_api_key):
        """Articles from all three pages are combined."""
        mock_get.side_effect = [
            _make_response(ARTICLE_PAGE_1, next_cursor="cursor-1"),
            _make_response(ARTICLE_PAGE_2, next_cursor="cursor-2"),
            _make_response(ARTICLE_PAGE_3, next_cursor=None),
        ]

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1 + ARTICLE_PAGE_2 + ARTICLE_PAGE_3)
        self.assertEqual(mock_get.call_count, 3)

    # ------------------------------------------------------------------
    # meta.next pagination shape
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_meta_next_pagination(self, mock_get, mock_kb_id, mock_api_key):
        """Handles responses that use meta.next instead of top-level next."""
        mock_get.side_effect = [
            _make_meta_response(ARTICLE_PAGE_1, next_cursor="meta-cursor-1"),
            _make_meta_response(ARTICLE_PAGE_2, next_cursor=None),
        ]

        result = self.module._fetch_all_articles()

        self.assertEqual(result, ARTICLE_PAGE_1 + ARTICLE_PAGE_2)
        self.assertEqual(mock_get.call_count, 2)

    # ------------------------------------------------------------------
    # Safety limit: stops after max_pages (10) even if more pages exist
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_safety_limit_stops_at_max_pages(self, mock_get, mock_kb_id, mock_api_key):
        """The loop stops after 10 pages even if the API keeps returning next cursors."""
        # Every page claims there is another page after it
        mock_get.side_effect = [
            _make_response([{"id": f"a{i}"}], next_cursor=f"cursor-{i}")
            for i in range(20)  # more than the 10-page cap
        ]

        result = self.module._fetch_all_articles()

        # Should have fetched exactly 10 articles (one per page, capped at 10)
        self.assertEqual(len(result), 10)
        self.assertEqual(mock_get.call_count, 10)

    # ------------------------------------------------------------------
    # Cache: second call returns cached result without extra HTTP requests
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_cache_prevents_duplicate_requests(
        self, mock_get, mock_kb_id, mock_api_key
    ):
        """Calling _fetch_all_articles() twice only hits the network once."""
        mock_get.return_value = _make_response(ARTICLE_PAGE_1, next_cursor=None)

        first = self.module._fetch_all_articles()
        second = self.module._fetch_all_articles()

        self.assertEqual(first, second)
        mock_get.assert_called_once()

    # ------------------------------------------------------------------
    # Empty data list (no articles at all)
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    @patch("src.tools.pylon_tools.requests.get")
    def test_empty_data_returns_empty_list(self, mock_get, mock_kb_id, mock_api_key):
        """An API response with no articles returns an empty list."""
        mock_get.return_value = _make_response([], next_cursor=None)

        result = self.module._fetch_all_articles()

        self.assertEqual(result, [])
        mock_get.assert_called_once()


class _ToolErrors(BaseCallbackHandler):
    def __init__(self):
        self.errors = []

    def on_tool_error(self, error, **kwargs):
        self.errors.append(error)


class TestPylonToolFailures(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import src.tools.pylon_tools as pylon_module

        self.module = pylon_module
        self.tool_errors = _ToolErrors()
        self.article = {
            "id": "a1",
            "title": "Public article",
            "is_published": True,
            "visibility_config": {"visibility": "public"},
            "identifier": "123",
            "slug": "public",
            "collection_id": "c1",
        }
        self.patches = [
            patch.object(pylon_module, "_articles_cache", None),
            patch.object(pylon_module, "_collections_cache", None),
            patch.object(pylon_module, "_get_api_key", return_value="secret-key"),
            patch.object(pylon_module, "_get_kb_id", return_value="private-kb"),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def invoke_tool(self, tool, args):
        middleware = ToolRetryMiddleware(initial_delay=0)
        node = ToolNode([tool], awrap_tool_call=middleware.awrap_tool_call)
        graph = StateGraph(MessagesState)
        graph.add_node("tools", node)
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        result = await graph.compile().ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": tool.name,
                                "args": args,
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            },
            config={"callbacks": [self.tool_errors]},
        )
        return result["messages"][-1]

    async def test_authorization_failures_are_errors_without_retries(self):
        for status in (401, 403):
            for tool, args in (
                (self.module.search_support_articles, {}),
                (self.module.get_support_article_content, {"article_id": "a1"}),
            ):
                for endpoint in ("articles", "collections"):
                    with self.subTest(status=status, tool=tool.name, endpoint=endpoint):
                        self.module._articles_cache = None
                        self.module._collections_cache = None
                        response = requests.Response()
                        response.status_code = status
                        response.url = (
                            "https://secret.invalid/private-kb?token=secret-key"
                        )
                        responses = [response]
                        if endpoint == "collections":
                            responses.insert(0, _make_response([self.article]))
                        with patch.object(
                            self.module.requests, "get", side_effect=responses
                        ) as get:
                            with self.assertLogs(
                                "src.middleware.tool_retry_middleware", level="WARNING"
                            ) as logs:
                                message = await self.invoke_tool(tool, args)
                        payload = json.loads(message.content)
                        self.assertEqual(message.status, "error")
                        self.assertEqual(
                            self.tool_errors.errors[-1].status_code, status
                        )
                        self.assertIn(
                            "Support KB unavailable", str(self.tool_errors.errors[-1])
                        )
                        self.assertEqual(payload["status_code"], status)
                        self.assertIn("Support KB unavailable", payload["details"])
                        self.assertIn("1 attempts", payload["message"])
                        self.assertEqual(get.call_count, len(responses))
                        for sensitive in ("secret-key", "private-kb", "secret.invalid"):
                            self.assertNotIn(
                                sensitive, message.content + str(logs.output)
                            )
                        self.assertIsNone(self.module._collections_cache)
                        if endpoint == "articles":
                            self.assertIsNone(self.module._articles_cache)

    async def test_transient_failures_retry_then_return_empty_success(self):
        for failure in (429, 503, "connection", "timeout"):
            with self.subTest(failure=failure):
                self.module._articles_cache = None
                if isinstance(failure, int):
                    response = requests.Response()
                    response.status_code = failure
                else:
                    response = (
                        requests.ConnectionError
                        if failure == "connection"
                        else requests.Timeout
                    )("sensitive URL")
                with patch.object(
                    self.module.requests,
                    "get",
                    side_effect=[response, _make_response([])],
                ) as get:
                    message = await self.invoke_tool(
                        self.module.search_support_articles, {}
                    )
                self.assertEqual(message.status, "success")
                self.assertEqual(json.loads(message.content)["total"], 0)
                self.assertEqual(get.call_count, 2)

    async def test_kb_failure_allows_parallel_documentation_read(self):
        @langchain_tool
        def read_official_docs() -> str:
            """Read official documentation."""
            return "Official documentation content"

        middleware = ToolRetryMiddleware(initial_delay=0)
        node = ToolNode(
            [self.module.search_support_articles, read_official_docs],
            awrap_tool_call=middleware.awrap_tool_call,
        )
        graph = StateGraph(MessagesState)
        graph.add_node("tools", node)
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        response = requests.Response()
        response.status_code = 401
        with patch.object(self.module.requests, "get", return_value=response):
            result = await graph.compile().ainvoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": name,
                                    "args": {},
                                    "id": name,
                                    "type": "tool_call",
                                }
                                for name in (
                                    "search_support_articles",
                                    "read_official_docs",
                                )
                            ],
                        )
                    ]
                }
            )
        kb, docs = result["messages"][-2:]
        self.assertEqual(kb.status, "error")
        self.assertIn("Support KB unavailable", kb.content)
        self.assertEqual(docs.status, "success")
        self.assertEqual(docs.content, "Official documentation content")

    async def test_failed_later_page_does_not_cache_partial_results(self):
        response = requests.Response()
        response.status_code = 403
        with patch.object(
            self.module.requests,
            "get",
            side_effect=[
                _make_response([self.article], next_cursor="private-cursor"),
                response,
            ],
        ):
            message = await self.invoke_tool(self.module.search_support_articles, {})
        self.assertEqual(message.status, "error")
        self.assertIsNone(self.module._articles_cache)
        self.assertNotIn("private-cursor", message.content)

    async def test_healthy_filtering_and_cache(self):
        articles = [
            self.article,
            dict(
                self.article, id="private", visibility_config={"visibility": "private"}
            ),
            dict(self.article, id="draft", is_published=False),
            dict(self.article, id="other", collection_id="c2"),
        ]
        collections = [
            {
                "id": "c1",
                "title": "General",
                "visibility_config": {"visibility": "public"},
            },
            {
                "id": "c3",
                "title": "Security",
                "visibility_config": {"visibility": "public"},
            },
        ]
        with patch.object(
            self.module.requests,
            "get",
            side_effect=[_make_response(articles), _make_response(collections)],
        ) as get:
            for _ in range(2):
                message = await self.invoke_tool(
                    self.module.search_support_articles, {"collections": "general"}
                )
                payload = json.loads(message.content)
                self.assertEqual(message.status, "success")
                self.assertEqual(
                    [article["id"] for article in payload["articles"]], ["a1"]
                )
                self.assertEqual(payload["articles"][0]["collection"], "General")
            empty = await self.invoke_tool(
                self.module.search_support_articles, {"collections": "Security"}
            )
            self.assertEqual(empty.status, "success")
            self.assertEqual(json.loads(empty.content)["total"], 0)
            self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
