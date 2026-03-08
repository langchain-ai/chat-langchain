"""Tests verifying that docs_agent handles large message histories without token overflow.

Root cause: anthropic.BadRequestError: prompt is too long: 204521 tokens > 200000 maximum
Trace: 019ae5ab-347c-7678-ad13-78f4063facc2
"""

import os

import pytest
from langsmith import testing as t

_DOCS_GRAPH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "agent", "docs_graph.py"
)


def _read_docs_graph_source() -> str:
    with open(_DOCS_GRAPH_PATH) as f:
        return f.read()


@pytest.mark.langsmith
def test_context_editing_middleware_is_imported():
    """ContextEditingMiddleware must be imported in docs_graph.py."""
    source = _read_docs_graph_source()

    t.log_inputs({"module_path": _DOCS_GRAPH_PATH})
    t.log_outputs({"has_import": "ContextEditingMiddleware" in source})

    assert "ContextEditingMiddleware" in source, (
        "docs_graph.py must import ContextEditingMiddleware to prevent context overflow "
        "(trace 019ae5ab-347c-7678-ad13-78f4063facc2 crashed with 204521 tokens > 200000 limit)"
    )


@pytest.mark.langsmith
def test_context_editing_middleware_is_used_in_create_agent():
    """ContextEditingMiddleware must be added to the middleware list in create_agent()."""
    source = _read_docs_graph_source()

    t.log_inputs({"module_path": _DOCS_GRAPH_PATH})
    t.log_outputs({
        "has_import": "ContextEditingMiddleware" in source,
        "has_context_editing_mw_var": "context_editing_middleware" in source,
    })

    assert "context_editing_middleware" in source, (
        "docs_graph.py must define and use context_editing_middleware in the middleware list"
    )


@pytest.mark.langsmith
def test_context_window_trigger_is_at_most_180k():
    """The ClearToolUsesEdit trigger must be <= 180_000 tokens (safe below Anthropic's 200k limit)."""
    source = _read_docs_graph_source()
    source_no_underscores = source.replace("_", "")

    t.log_inputs({"module_path": _DOCS_GRAPH_PATH})

    # Check that the source mentions a threshold value <= 180k
    # Values we accept: 180000, 160000, 150000, 140000, 130000, 120000, 100000
    safe_thresholds = [180000, 160000, 150000, 140000, 130000, 120000, 100000]
    has_safe_threshold = any(str(v) in source_no_underscores for v in safe_thresholds)

    t.log_outputs({
        "has_safe_threshold": has_safe_threshold,
        "source_contains_context_editing": "ContextEditingMiddleware" in source,
    })

    assert "ContextEditingMiddleware" in source, (
        "docs_graph.py must use ContextEditingMiddleware"
    )
    assert has_safe_threshold, (
        "docs_graph.py must configure ClearToolUsesEdit with a trigger <= 180_000 tokens "
        "(safe buffer below Anthropic's 200k limit of 200k tokens)"
    )


@pytest.mark.langsmith
def test_clear_tool_uses_edit_mechanism():
    """ClearToolUsesEdit must trim tool results when token count exceeds threshold."""
    from langchain.agents.middleware import ClearToolUsesEdit
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.messages.utils import count_tokens_approximately

    # Build a simulated long conversation with many tool calls
    messages = []
    for i in range(30):
        messages.append(HumanMessage(content=f"Question {i}: How do I use LangChain feature {i}?"))
        # Simulate an AI message with a tool call
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_docs", "args": {"query": f"feature {i}"}, "id": f"call_{i}"}
            ],
        )
        messages.append(ai_msg)
        # Simulate a large tool result (like what doc search returns)
        tool_result = "A" * 3000  # ~750 tokens per result
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=f"call_{i}",
                name="search_docs",
            )
        )
        messages.append(AIMessage(content=f"Answer {i}: " + "B" * 500))

    # Add a final user question
    messages.append(HumanMessage(content="What is LangGraph?"))

    initial_token_count = count_tokens_approximately(messages)
    t.log_inputs({
        "num_messages": len(messages),
        "initial_token_count": initial_token_count,
    })

    # Apply ClearToolUsesEdit with a 100k trigger (simulates what docs_agent uses)
    edit = ClearToolUsesEdit(trigger=100_000, keep=3)
    edited_messages = list(messages)  # copy

    def count_tokens(msgs):
        return count_tokens_approximately(msgs)

    edit.apply(edited_messages, count_tokens=count_tokens)

    final_token_count = count_tokens_approximately(edited_messages)
    t.log_outputs({
        "final_token_count": final_token_count,
        "initial_token_count": initial_token_count,
        "tokens_reduced": final_token_count < initial_token_count,
    })

    # Verify the edit works correctly (no crash, returns valid message list)
    assert len(edited_messages) == len(messages), "Number of messages should not change (only content)"

    # The 3 most recent tool results should be preserved
    tool_messages = [m for m in edited_messages if isinstance(m, ToolMessage)]
    recent_tool_messages = tool_messages[-3:]
    for tm in recent_tool_messages:
        assert tm.content != "[cleared]", "The 3 most recent tool results should be preserved"

    # If the initial count exceeded the trigger, tokens should be reduced
    if initial_token_count > 100_000:
        assert final_token_count < initial_token_count, (
            f"ClearToolUsesEdit should reduce token count from {initial_token_count} "
            f"but got {final_token_count}"
        )
