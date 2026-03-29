"""Unit tests for per-invocation deduplication in SearchDocsByLangChain.

The fix adds task-keyed deduplication so that calling SearchDocsByLangChain
with the same (query, page_size, language) tuple more than once within the
same async invocation context returns an "already retrieved" message on the
second (and subsequent) call, without hitting the API or cache.
"""

from unittest.mock import patch

import pytest

import src.tools.docs_tools as docs_tools_module
from src.tools.docs_tools import SearchDocsByLangChain


def _reset_dedup_state():
    """Clear the per-invocation dedup state so sync tests are independent."""
    # Reset the thread-local fallback used in sync (non-async) contexts
    if hasattr(docs_tools_module._thread_searched, "keys"):
        docs_tools_module._thread_searched.keys = set()


# ---------------------------------------------------------------------------
# 1. Sync context: second call with identical args returns "already retrieved"
# ---------------------------------------------------------------------------

def test_second_call_same_query_returns_already_retrieved_message():
    """Second SearchDocsByLangChain call with identical args must short-circuit."""
    _reset_dedup_state()

    fake_result = "Result 1:\nTitle: Streaming Guide\nLink: https://docs.langchain.com/streaming\nContent: ..."

    with patch.object(docs_tools_module, "_get_from_cache_fuzzy", return_value=fake_result) as mock_cache, \
         patch.object(docs_tools_module, "_search_docs_api") as mock_api:

        # First call — should go through normally
        result1 = SearchDocsByLangChain.invoke({"query": "streaming", "page_size": 5, "language": "python"})

        # Second call — identical args, should be blocked
        result2 = SearchDocsByLangChain.invoke({"query": "streaming", "page_size": 5, "language": "python"})

    # First call hit the cache
    assert mock_cache.call_count == 1
    # API was never called (cache hit)
    mock_api.assert_not_called()

    # First result is the real result
    assert result1 == fake_result

    # Second result is the dedup message
    assert "[Already retrieved]" in result2
    assert "streaming" in result2
    assert "conversation context" in result2


def test_second_call_does_not_invoke_api():
    """Dedup must prevent the API from being called on duplicate invocations."""
    _reset_dedup_state()

    with patch.object(docs_tools_module, "_get_from_cache_fuzzy", return_value=None), \
         patch.object(docs_tools_module, "_search_docs_api", return_value="some result") as mock_api, \
         patch.object(docs_tools_module, "_store_in_cache"):

        SearchDocsByLangChain.invoke({"query": "middleware", "page_size": 5, "language": "python"})
        SearchDocsByLangChain.invoke({"query": "middleware", "page_size": 5, "language": "python"})

    # API should only have been called once — the second call is blocked before the API
    assert mock_api.call_count == 1


# ---------------------------------------------------------------------------
# 2. Different queries in same context are NOT blocked
# ---------------------------------------------------------------------------

def test_different_queries_are_not_deduplicated():
    """Different queries within the same context should each be allowed through."""
    _reset_dedup_state()

    fake_result = "Result 1:\nTitle: Doc\nLink: https://docs.langchain.com/x\nContent: ..."

    with patch.object(docs_tools_module, "_get_from_cache_fuzzy", return_value=fake_result) as mock_cache:

        result_streaming = SearchDocsByLangChain.invoke({"query": "streaming", "page_size": 5, "language": "python"})
        result_middleware = SearchDocsByLangChain.invoke({"query": "middleware", "page_size": 5, "language": "python"})

    # Both should hit the cache (not deduped)
    assert mock_cache.call_count == 2
    assert "[Already retrieved]" not in result_streaming
    assert "[Already retrieved]" not in result_middleware


# ---------------------------------------------------------------------------
# 3. Dedup is isolated per async task (no cross-invocation leakage)
# ---------------------------------------------------------------------------

def test_new_async_task_starts_fresh():
    """Each new asyncio task has its own dedup state — no cross-invocation leakage."""
    import asyncio

    fake_result = "Result 1:\nTitle: Agents\nLink: https://docs.langchain.com/agents\nContent: ..."

    results = []

    async def run_as_task():
        """Simulate a fresh agent invocation in a new async task."""
        with patch.object(docs_tools_module, "_get_from_cache_fuzzy", return_value=fake_result):
            result = SearchDocsByLangChain.invoke({"query": "agents", "page_size": 5, "language": "python"})
            results.append(result)

    async def main():
        # Run two independent tasks — each should see "agents" as not-yet-searched
        await asyncio.gather(
            asyncio.ensure_future(run_as_task()),
            asyncio.ensure_future(run_as_task()),
        )

    asyncio.run(main())

    # Both tasks should have gotten real results, not the dedup message
    assert len(results) == 2
    for r in results:
        assert "[Already retrieved]" not in r


def test_dedup_works_within_single_async_task():
    """Within the same asyncio task, duplicate queries ARE blocked."""
    import asyncio

    fake_result = "Result 1:\nTitle: Streaming\nLink: https://docs.langchain.com/streaming\nContent: ..."

    async def run():
        # Ensure a fresh task-local state for the current task
        task = asyncio.current_task()
        if task in docs_tools_module._task_searched:
            del docs_tools_module._task_searched[task]

        with patch.object(docs_tools_module, "_get_from_cache_fuzzy", return_value=fake_result):
            r1 = SearchDocsByLangChain.invoke({"query": "streaming", "page_size": 5, "language": "python"})
            r2 = SearchDocsByLangChain.invoke({"query": "streaming", "page_size": 5, "language": "python"})
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert "[Already retrieved]" not in r1
    assert "[Already retrieved]" in r2


# ---------------------------------------------------------------------------
# 4. Abbreviation normalization: "auth" and "authentication" treated as same key
# ---------------------------------------------------------------------------

def test_normalized_query_deduplication():
    """Queries that normalize to the same string should be deduplicated."""
    _reset_dedup_state()

    fake_result = "Result 1:\nTitle: Auth\nLink: https://docs.langchain.com/auth\nContent: ..."

    with patch.object(docs_tools_module, "_get_from_cache_fuzzy", return_value=fake_result) as mock_cache:

        # "auth" normalizes to "authentication"
        result1 = SearchDocsByLangChain.invoke({"query": "auth", "page_size": 5, "language": "python"})
        # "authentication" also normalizes to "authentication" — should be deduped
        result2 = SearchDocsByLangChain.invoke({"query": "authentication", "page_size": 5, "language": "python"})

    assert mock_cache.call_count == 1
    assert result1 == fake_result
    assert "[Already retrieved]" in result2
