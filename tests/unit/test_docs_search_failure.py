"""Tests for SearchDocsByLangChain retry-exhaustion behavior.

P0 Bug: `SearchDocsByLangChain` previously caught retry exhaustion and returned
a stringified JSON error blob (`{"error": "Documentation search unavailable",
...}`) as a successful tool result. The agent framework treated this as a
normal response, so the LLM never realized docs retrieval failed and would
fabricate stale-shape `docs.langchain.com/...` citation URLs from parametric
knowledge.

Fix: raise `RuntimeError` instead. The agent framework wraps raised exceptions
as `ToolMessage(status="error")`, which the LLM distinguishes from successful
tool results.

Root cause: src/tools/docs_tools.py retry-exhaustion path (~lines 308-315).
"""

from unittest.mock import patch

import pytest

from src.tools import docs_tools
from src.tools.docs_tools import SearchDocsByLangChain


def test_retry_exhaustion_raises_runtime_error(monkeypatch):
    """When `_search_docs_api` always raises, the tool must raise RuntimeError
    (not return a JSON error string)."""

    def _always_raise(*args, **kwargs):
        raise ConnectionError("upstream Mintlify down")

    monkeypatch.setattr(docs_tools, "_search_docs_api", _always_raise)
    # Avoid sleeping between retries during the test.
    monkeypatch.setattr(docs_tools.time, "sleep", lambda _s: None)
    # Bypass cache so we exercise the retry path.
    monkeypatch.setattr(docs_tools, "_get_from_cache_fuzzy", lambda *a, **kw: None)
    monkeypatch.setattr(docs_tools, "_store_in_cache", lambda *a, **kw: None)

    with pytest.raises(RuntimeError) as excinfo:
        SearchDocsByLangChain.invoke({"query": "anything"})

    msg = str(excinfo.value)
    assert "SearchDocsByLangChain unavailable" in msg
    assert "upstream Mintlify down" in msg


def test_retry_exhaustion_does_not_return_json_error_blob(monkeypatch):
    """Regression: the old code returned `json.dumps({"error": "Documentation
    search unavailable", ...})` instead of raising. Ensure that exact
    behaviour does not return; the tool must raise."""

    def _always_raise(*args, **kwargs):
        raise ConnectionError("boom")

    monkeypatch.setattr(docs_tools, "_search_docs_api", _always_raise)
    monkeypatch.setattr(docs_tools.time, "sleep", lambda _s: None)
    monkeypatch.setattr(docs_tools, "_get_from_cache_fuzzy", lambda *a, **kw: None)
    monkeypatch.setattr(docs_tools, "_store_in_cache", lambda *a, **kw: None)

    with pytest.raises(RuntimeError):
        # If the bug regressed, .invoke would return a string instead of raising.
        SearchDocsByLangChain.invoke({"query": "middleware"})


def test_cache_hit_path_unaffected(monkeypatch):
    """A cache hit must still return the cached string without invoking the
    API or raising — the fix only changes the retry-exhaustion path."""

    monkeypatch.setattr(
        docs_tools,
        "_get_from_cache_fuzzy",
        lambda *a, **kw: "cached results",
    )

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("_search_docs_api must not be called on cache hit")

    monkeypatch.setattr(docs_tools, "_search_docs_api", _should_not_be_called)

    result = SearchDocsByLangChain.invoke({"query": "middleware"})
    assert result == "cached results"
