"""Tests that _format_search_results truncates content to prevent context overflow.

Root cause: No truncation applied to doc page content, allowing single results
up to 53k chars that cause BadRequestError (prompt too long) when accumulated.
"""
import pytest
from langsmith import testing as t, expect

from src.tools.docs_tools import _format_search_results, MAX_CONTENT_CHARS


@pytest.mark.langsmith
def test_format_search_results_truncates_very_long_content():
    """Long doc content must be truncated to prevent context window overflow."""
    # Simulate a Mintlify result with massive content (like the 53k-char real cases)
    long_content = "A" * 100_000
    results = [
        {
            "metadata": {"title": "LangGraph Streaming Guide"},
            "path": "/oss/python/langgraph/streaming",
            "content": long_content,
        }
    ]
    t.log_inputs({"content_length": len(long_content), "num_results": 1})

    output = _format_search_results(results)
    t.log_outputs({"output_length": len(output)})
    t.log_reference_outputs({"max_output_length": MAX_CONTENT_CHARS + 500})  # +500 for metadata overhead

    assert len(output) < MAX_CONTENT_CHARS + 500, (
        f"Output {len(output)} chars exceeds limit {MAX_CONTENT_CHARS + 500}. "
        "Content must be truncated to prevent context window overflow."
    )
    assert "LangGraph Streaming Guide" in output, "Title must be preserved after truncation"
    assert "truncated" in output.lower() or len(output) <= MAX_CONTENT_CHARS + 500


@pytest.mark.langsmith
def test_format_search_results_preserves_short_content():
    """Short content must not be truncated."""
    short_content = "This is a short description of the LangGraph streaming API."
    results = [
        {
            "metadata": {"title": "Quick Overview"},
            "path": "/oss/python/langgraph/overview",
            "content": short_content,
        }
    ]
    t.log_inputs({"content_length": len(short_content)})

    output = _format_search_results(results)
    t.log_outputs({"output_length": len(output)})

    assert short_content in output, "Short content must be preserved unchanged"


@pytest.mark.langsmith
def test_format_search_results_bounds_total_size_with_multiple_results():
    """Total output with multiple long results must stay within safe bounds."""
    # Simulate 5 results with massive content (page_size=5 default)
    results = [
        {
            "metadata": {"title": f"Doc Page {i}"},
            "path": f"/path/to/doc{i}",
            "content": "X" * 60_000,  # 60k chars per result, like the 53k real cases
        }
        for i in range(5)
    ]
    t.log_inputs({"num_results": 5, "content_per_result": 60_000, "total_raw": 300_000})

    output = _format_search_results(results)
    total_len = len(output)
    t.log_outputs({"total_output_length": total_len})
    t.log_reference_outputs({"max_safe_length": 5 * (MAX_CONTENT_CHARS + 200)})

    max_safe = 5 * (MAX_CONTENT_CHARS + 200)  # 200 chars overhead per result for metadata
    assert total_len <= max_safe, (
        f"5 results produced {total_len:,} chars, exceeding safe limit of {max_safe:,}. "
        "Content truncation needed to prevent context overflow."
    )


@pytest.mark.langsmith
def test_max_content_chars_constant_is_defined_and_reasonable():
    """MAX_CONTENT_CHARS must exist and be a reasonable limit (1000-15000)."""
    t.log_inputs({"check": "MAX_CONTENT_CHARS constant"})
    t.log_outputs({"MAX_CONTENT_CHARS": MAX_CONTENT_CHARS})
    t.log_reference_outputs({"min": 1000, "max": 15000})

    assert isinstance(MAX_CONTENT_CHARS, int), "MAX_CONTENT_CHARS must be an integer"
    assert 1000 <= MAX_CONTENT_CHARS <= 15000, (
        f"MAX_CONTENT_CHARS={MAX_CONTENT_CHARS} is outside reasonable range [1000, 15000]. "
        "Too small loses information; too large allows context overflow."
    )
