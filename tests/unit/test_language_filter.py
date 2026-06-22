"""Tests for the language post-filter in _format_search_results.

Bug: SearchDocsByLangChain returned /oss/javascript/ URLs even when
language="python" (the default) was requested, because the upstream Mintlify
index has gaps in Python coverage and falls back to JavaScript pages instead
of returning empty.

Fix: src/tools/docs_tools.py — _format_search_results now drops results whose
link/path is for the wrong language before formatting, and emits a clear
"no docs in this language" message when the filter empties the list.
"""

from unittest.mock import patch

from src.tools.docs_tools import _format_search_results


def _result(path: str, title: str = "Doc", content: str = "content") -> dict:
    return {
        "path": path,
        "metadata": {"title": title},
        "content": content,
    }


def _result_with_link(link: str, title: str = "Doc", content: str = "content") -> dict:
    return {
        "link": link,
        "metadata": {"title": title},
        "content": content,
    }


class TestLanguageFilter:
    def test_python_request_drops_javascript_only_results(self):
        results = [
            _result_with_link(
                "https://docs.langchain.com/oss/javascript/langgraph/interrupts"
            )
        ]
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results(results, language="python")
        assert "/oss/javascript/" not in output
        assert "No python-language documentation found" in output

    def test_python_request_keeps_python_drops_javascript_mixed(self):
        results = [
            _result("/oss/python/langgraph/streaming", title="Streaming PY"),
            _result("/oss/javascript/langgraph/streaming", title="Streaming JS"),
        ]
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results(results, language="python")
        assert "/oss/python/langgraph/streaming" in output
        assert "/oss/javascript/" not in output
        assert "Streaming PY" in output
        assert "Streaming JS" not in output

    def test_default_language_is_python(self):
        results = [
            _result_with_link(
                "https://docs.langchain.com/oss/javascript/langgraph/interrupts"
            )
        ]
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results(results)
        assert "/oss/javascript/" not in output
        assert "No python-language documentation found" in output

    def test_javascript_request_drops_python_only_results(self):
        results = [
            _result("/oss/python/langgraph/streaming", title="Streaming PY"),
        ]
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results(results, language="javascript")
        assert "/oss/python/" not in output
        assert "No javascript-language documentation found" in output

    def test_empty_results_still_returns_no_results_message(self):
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([], language="python")
        assert output == "No results found."
