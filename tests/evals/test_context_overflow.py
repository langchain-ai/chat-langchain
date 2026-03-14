# tests/evals/test_context_overflow.py
"""Tests to verify that context overflow is prevented.

Traces showing the crash:
- https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/projects/p/dcffe24f-52f0-434f-aa22-932d27cb23ef/r/019ae5ab-347c-7678-ad13-78f4063facc2
- https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/projects/p/dcffe24f-52f0-434f-aa22-932d27cb23ef/r/019ae55f-f904-7127-bb02-d1bffbf9ec5d
- https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/projects/p/dcffe24f-52f0-434f-aa22-932d27cb23ef/r/019ae4c9-30c2-7403-853d-28af82ef62c7
"""

import pytest
from langchain.agents.middleware import ContextEditingMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langsmith import testing as t


@pytest.mark.langsmith
def test_context_editing_middleware_is_exported_from_docs_graph():
    """ContextEditingMiddleware must be exported from docs_graph module.

    Without it, long conversations exceed the 200k token limit and crash.
    """
    import src.agent.docs_graph as dg

    t.log_inputs({"check": "context_editing_middleware in docs_graph module"})

    has_middleware = hasattr(dg, "context_editing_middleware")
    t.log_outputs({"has_context_editing_middleware": has_middleware})
    t.log_reference_outputs({"expected": True})

    assert has_middleware, (
        "context_editing_middleware not found in src.agent.docs_graph. "
        "It must be created and wired into docs_agent's middleware list."
    )

    cem = dg.context_editing_middleware
    assert isinstance(cem, ContextEditingMiddleware), (
        f"context_editing_middleware is {type(cem)}, expected ContextEditingMiddleware"
    )


@pytest.mark.langsmith
def test_context_editing_middleware_trigger_is_below_200k():
    """ContextEditingMiddleware must trigger well below the 200k token limit.

    The trigger threshold must be low enough that token pruning fires before
    the conversation hits the model's 200k hard limit.
    """
    import src.agent.docs_graph as dg

    t.log_inputs({"check": "ContextEditingMiddleware trigger < 200000"})

    assert hasattr(dg, "context_editing_middleware"), (
        "context_editing_middleware not found in src.agent.docs_graph"
    )
    cem = dg.context_editing_middleware
    assert isinstance(cem, ContextEditingMiddleware)

    # Every ClearToolUsesEdit trigger must be below 200k
    for edit in cem.edits:
        if isinstance(edit, ClearToolUsesEdit):
            t.log_outputs({"trigger": edit.trigger})
            t.log_reference_outputs({"max_trigger": 180_000})
            assert edit.trigger < 200_000, (
                f"ClearToolUsesEdit trigger={edit.trigger} is too high. "
                f"Must be < 200000 to avoid BadRequestError."
            )


@pytest.mark.langsmith
def test_context_editing_middleware_in_docs_agent_create_call():
    """context_editing_middleware must be included in the docs_agent middleware list.

    Verify it is present alongside guardrails, retry, and fallback middleware.
    """
    import src.agent.docs_graph as dg

    t.log_inputs({"check": "context_editing_middleware wired into docs_agent"})

    # The docs_graph module should expose all middleware objects used by the agent
    required_middleware_attrs = [
        "guardrails_middleware",
        "model_retry_middleware",
        "model_fallback_middleware",
        "context_editing_middleware",
    ]

    missing = [attr for attr in required_middleware_attrs if not hasattr(dg, attr)]
    t.log_outputs({"missing_middleware": missing})
    t.log_reference_outputs({"missing_middleware": []})

    assert not missing, (
        f"Missing middleware in docs_graph: {missing}. "
        f"All middleware must be module-level variables wired into create_agent()."
    )

    # Confirm it's the right type
    cem = dg.context_editing_middleware
    assert isinstance(cem, ContextEditingMiddleware), (
        f"context_editing_middleware is {type(cem)}, expected ContextEditingMiddleware"
    )


@pytest.mark.langsmith
def test_clear_tool_uses_edit_reduces_tokens_when_triggered():
    """ClearToolUsesEdit must reduce token count when the threshold is exceeded.

    This is a unit test of the pruning logic itself to confirm that large
    tool results get cleared when the conversation grows too long.
    """
    BIG_CONTENT = "x" * 50_000  # ~12k tokens each
    # Build a conversation with many large tool messages
    messages = []
    for i in range(5):
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"call_{i}",
                    "name": "SearchDocsByLangChain",
                    "args": {"query": "streaming"},
                }
            ],
        )
        tool_msg = ToolMessage(
            content=BIG_CONTENT,
            tool_call_id=f"call_{i}",
            name="SearchDocsByLangChain",
        )
        messages.append(ai_msg)
        messages.append(tool_msg)
    messages.append(HumanMessage(content="What is LangChain?"))

    tokens_before = count_tokens_approximately(messages)
    t.log_inputs({"tokens_before": tokens_before, "message_count": len(messages)})

    # Apply the edit with a trigger well below the total token count
    edit = ClearToolUsesEdit(trigger=10_000, keep=1)
    edit.apply(messages, count_tokens=count_tokens_approximately)

    tokens_after = count_tokens_approximately(messages)
    t.log_outputs({"tokens_before": tokens_before, "tokens_after": tokens_after})
    t.log_reference_outputs({"tokens_reduced": True})

    assert tokens_after < tokens_before, (
        f"ClearToolUsesEdit did not reduce token count: "
        f"before={tokens_before}, after={tokens_after}"
    )
    assert tokens_after < tokens_before * 0.9, (
        f"ClearToolUsesEdit reduced by less than 10%: "
        f"before={tokens_before}, after={tokens_after}"
    )


@pytest.mark.langsmith
def test_clear_tool_uses_edit_keeps_recent_results():
    """ClearToolUsesEdit must preserve the most recent tool results (keep=N).

    The 'keep' parameter ensures the latest tool results stay in context,
    so the agent can still reference recent search output.
    """
    BIG_CONTENT = "y" * 20_000
    messages = []
    for i in range(4):
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"call_{i}",
                    "name": "SearchDocsByLangChain",
                    "args": {"query": f"topic_{i}"},
                }
            ],
        )
        tool_msg = ToolMessage(
            content=f"Result for topic_{i}: " + BIG_CONTENT,
            tool_call_id=f"call_{i}",
            name="SearchDocsByLangChain",
        )
        messages.append(ai_msg)
        messages.append(tool_msg)

    keep = 2
    edit = ClearToolUsesEdit(trigger=1_000, keep=keep)
    edit.apply(messages, count_tokens=count_tokens_approximately)

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    preserved = [m for m in tool_messages if m.content != "[cleared]"]
    cleared = [m for m in tool_messages if m.content == "[cleared]"]

    t.log_outputs(
        {
            "total_tool_messages": len(tool_messages),
            "preserved": len(preserved),
            "cleared": len(cleared),
        }
    )
    t.log_reference_outputs({"preserved": keep})

    assert len(preserved) == keep, (
        f"Expected {keep} preserved tool messages, got {len(preserved)}. "
        f"ClearToolUsesEdit(keep={keep}) should keep the {keep} most recent results."
    )


@pytest.mark.langsmith
def test_clear_tool_uses_edit_noop_below_trigger():
    """ClearToolUsesEdit must not alter messages when under the token threshold."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_0",
                    "name": "SearchDocsByLangChain",
                    "args": {"query": "agents"},
                }
            ],
        ),
        ToolMessage(
            content="Short result", tool_call_id="call_0", name="SearchDocsByLangChain"
        ),
        HumanMessage(content="Thanks"),
    ]
    original_contents = [m.content for m in messages]

    edit = ClearToolUsesEdit(trigger=200_000, keep=1)  # Very high trigger
    edit.apply(messages, count_tokens=count_tokens_approximately)

    contents_after = [m.content for m in messages]
    t.log_inputs({"tokens": count_tokens_approximately(messages)})
    t.log_outputs({"messages_unchanged": contents_after == original_contents})
    t.log_reference_outputs({"messages_unchanged": True})

    assert contents_after == original_contents, (
        "ClearToolUsesEdit modified messages even though token count is below trigger threshold."
    )
