"""Tests for broken documentation URL construction in _format_search_results.

P0 Bug: URLs were constructed as f"https://docs.langchain.com{path}" without
ensuring a slash separator. Paths returned by Mintlify without a leading slash
(e.g., "oss/python/langgraph/streaming") produced invalid URLs like
"https://docs.langchain.comoss/python/langgraph/streaming".

Root cause: src/tools/docs_tools.py line 222
Fix: use path.lstrip('/') with an explicit slash in the f-string.
"""

from unittest.mock import patch

import pytest

from src.tools.docs_tools import _format_search_results

BASE_URL = "https://docs.langchain.com"


# ------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_result(path: str, title: str = "Test Page", content: str = "Some content") -> dict:
    """Return a minimal Mintlify search result dict."""
    return {
        "path": path,
        "metadata": {"title": title},
        "content": content,
    }


def _extract_url_from_output(output: str) -> str:
    """Pull the 'Link: <url>' value out of _format_search_results output."""
    for line in output.splitlines():
        if line.startswith("Link:"):
            return line.split("Link:", 1)[1].strip()
    raise ValueError(f"No 'Link:' line found in output:\n{output}")


# ------------------------------------------------------------------------
# Core URL-construction tests
# ------------------------------------------------------------------------


class TestUrlConstructionWithoutLeadingSlash:
    """Paths returned WITHOUT a leading slash must still produce valid URLs."""

    @pytest.mark.parametrize(
        "path",
        [
            "oss/python/langgraph/streaming",
            "oss/python/langchain/tools",
            "oss/python/langgraph/concepts/streaming",
            "langsmith/concepts/tracing",
        ],
    )
    def test_url_has_slash_separator(self, path):
        """URL must contain a '/' between the domain and the path segment."""
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(path)])
        url = _extract_url_from_output(output)
        expected = f"{BASE_URL}/{path}"
        assert url == expected, (
            f"Expected '{expected}', got '{url}'. "
            "Missing slash between domain and path — the P0 URL bug is present."
        )

    def test_url_does_not_merge_domain_and_path(self):
        """Regression: domain must not be merged with path (the original bug)."""
        path = "oss/python/langgraph/streaming"
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(path)])
        url = _extract_url_from_output(output)
        assert "https://docs.langchain.comoss" not in url, (
            "P0 bug detected: domain and path are merged without a slash separator."
        )

    def test_url_starts_with_base_url_slash(self):
        """URL must start with 'https://docs.langchain.com/'."""
        path = "oss/python/langchain/introduction"
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(path)])
        url = _extract_url_from_output(output)
        assert url.startswith(
            f"{BASE_URL}/"
        ), (
            f"URL '{url}' does not start with ';BASE_URL}/'."
        )


class TestUrlConstructionWithLeadingSlash:
    """Paths returned WITH a leading slash must not produce double-slashes."""

    @pytest.mark.parametrize(
        "path",
        [
            "/oss/python/langgraph/streaming",
            "/oss/python/langchain/tools",
            "/langsmith/concepts/tracing",
        ],
    )
    def test_no_double_slash(self, path):
        """URL must not contain '//' after the protocol."""
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(path)])
        url = _extract_url_from_output(output)
        # Strip the protocol-level '//' before checking
        url_without_protocol = url[len("https://"):]
        assert "//" not in url_without_protocol, (
            f"Double-slash detected in URL '{url}'."
        )

    def test_leading_slash_path_normalised(self):
        """A path with a leading slash must produce the same URL as one without."""
        path_with_slash = "/oss/python/langgraph/streaming"
        path_without_slash = "oss/python/langgraph/streaming"
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            out_with = _format_search_results([_make_result(path_with_slash)])
            out_without = _format_search_results([_make_result(path_without_slash)])
        url_with = _extract_url_from_output(out_with)
        url_without = _extract_url_from_output(out_without)
        assert url_with == url_without, (
            f"Leading-slash path produced '{url_with}', "
            f"no-slash path produced '{url_without}'. They should be identical."
        )


# ------------------------------------------------------------------------
# _format_search_results integration tests
# ------------------------------------------------------------------------


class TestFormatSearchResults:
    """Broader tests for _format_search_results URL handling."""

    def test_empty_path_returns_na(self):
        """An empty path must produce 'N/A' for the link."""
        result = _make_result(path="")
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([result])
        url = _extract_url_from_output(output)
        assert url == "N/A"

    def test_none_path_fallback(self):
        """A missing 'path' key must produce 'N/A' for the link."""
        result = {"metadata": {"title": "No Path"}, "content": "content"}
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([result])
        url = _extract_url_from_output(output)
        assert url == "N/A"

    def test_multiple_results_all_valid_urls(self):
        """Every result in a multi-result response must have a valid URL."""
        results = [
            _make_result("oss/python/langgraph/streaming", title="Streaming"),
            _make_result("/oss/python/langchain/tools", title="Tools"),
            _make_result("langsmith/evaluation/quickstart", title="Eval Quickstart"),
        ]
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results(results)

        urls = [
            line.split("Link:", 1)[1].strip()
            for line in output.splitlines()
            if line.startswith("Link:")
        ]
        assert len(urls) == 3, f"Expected 3 URLs, got {len(urls)}"
        for url in urls:
            assert url.startswith(f"{BASE_URL}/"), (
                f"URL '{url}' does not start with '{BASE_URL}/'."
            )
            url_without_protocol = url[len("https://"):]
            assert "//" not in url_without_protocol, (
                f"Double-slash in URL '{url}'."
            )

    def test_no_results_returns_no_results_message(self):
        """An empty results list must return the 'No results found.' message."""
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([])
        assert output == "No results found."

    def test_real_world_langgraph_streaming_url(self):
        """Reproduce the exact production trace path that was 404-ing."""
        path = "oss/python/langgraph/streaming"  # no leading slash — production case
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(path)])
        url = _extract_url_from_output(output)
        assert url == "https://docs.langchain.com/oss/python/langgraph/streaming", (
            f"Got '{url}' — the production P0 URL bug is still present."
        )
