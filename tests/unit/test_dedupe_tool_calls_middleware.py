"""Tests for DedupeToolCallsMiddleware.

The docs agent occasionally loops on `(tool_name, args)` repeats within one
turn — retrying the same `.mdx` file with `rg` → `grep` → `head` → `cat`, or
firing synonym/case restatements of the same search. This middleware
short-circuits the second occurrence of a `(tool_name, canonical_args)`
signature with a synthetic ToolMessage so the LLM still gets feedback but
the underlying tool isn't re-invoked.

These tests pin the contract:
  1. First call → handler runs, result passes through.
  2. Identical second call → handler is NOT invoked, synthetic ToolMessage
     is returned, signature names the offending tool/args.
  3. Different args → handler runs again (not a duplicate).
  4. New HumanMessage → turn boundary advances, prior signatures clear.
"""
from __future__ import annotations

import pytest

# These imports trigger third-party module loads at collection time; if the
# environment can't satisfy them the whole test module is skipped rather
# than failing. CI installs the package dependencies so the real assertion
# path always runs there.
pytest.importorskip("langchain.agents.middleware")
pytest.importorskip("langgraph.prebuilt.tool_node")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.middleware.dedupe_tool_calls_middleware import (  # noqa: E402
    DedupeToolCallsMiddleware,
)


class _FakeRequest:
    """Stand-in for `langgraph.prebuilt.tool_node.ToolCallRequest`.

    The real type is a dataclass with `tool_call`, `state`, and `runtime`
    attributes. The middleware only touches `tool_call` and `state`, so we
    mirror just those.
    """

    def __init__(self, name: str, args: dict, *, state: dict, call_id: str = "c1"):
        self.tool_call = {"name": name, "args": args, "id": call_id}
        self.state = state
        self.runtime = None


def _state_with_messages(*messages) -> dict:
    return {"messages": list(messages)}


@pytest.mark.asyncio
async def test_duplicate_call_is_short_circuited():
    """Second identical call returns synthetic ToolMessage; handler not invoked."""
    middleware = DedupeToolCallsMiddleware()
    state = _state_with_messages(HumanMessage(content="What is middleware?"))

    invocations = 0

    async def handler(req):
        nonlocal invocations
        invocations += 1
        return ToolMessage(
            content="middleware results",
            name=req.tool_call["name"],
            tool_call_id=req.tool_call["id"],
        )

    req1 = _FakeRequest(
        "search_docs_by_lang_chain", {"query": "middleware"}, state=state, call_id="a"
    )
    req2 = _FakeRequest(
        "search_docs_by_lang_chain", {"query": "middleware"}, state=state, call_id="b"
    )

    first = await middleware.awrap_tool_call(req1, handler)
    second = await middleware.awrap_tool_call(req2, handler)

    # The first call ran the real handler; the second was short-circuited.
    assert invocations == 1
    assert isinstance(first, ToolMessage)
    assert first.content == "middleware results"

    assert isinstance(second, ToolMessage)
    assert "already made on this turn" in second.content
    assert second.name == "search_docs_by_lang_chain"
    # The synthetic message must carry the new tool_call_id so the LLM can
    # match it to the duplicate call it just issued.
    assert second.tool_call_id == "b"


@pytest.mark.asyncio
async def test_different_args_are_not_deduplicated():
    """Same tool with different args is a NEW call, not a duplicate."""
    middleware = DedupeToolCallsMiddleware()
    state = _state_with_messages(HumanMessage(content="docs?"))

    invocations = 0

    async def handler(req):
        nonlocal invocations
        invocations += 1
        return ToolMessage(
            content="ok", name=req.tool_call["name"], tool_call_id=req.tool_call["id"]
        )

    await middleware.awrap_tool_call(
        _FakeRequest("search_docs_by_lang_chain", {"query": "tracing"}, state=state),
        handler,
    )
    await middleware.awrap_tool_call(
        _FakeRequest("search_docs_by_lang_chain", {"query": "callbacks"}, state=state),
        handler,
    )

    assert invocations == 2


@pytest.mark.asyncio
async def test_arg_key_order_does_not_break_dedup():
    """Canonical JSON sorts keys, so {a:1,b:2} == {b:2,a:1} for dedup."""
    middleware = DedupeToolCallsMiddleware()
    state = _state_with_messages(HumanMessage(content="x"))

    invocations = 0

    async def handler(req):
        nonlocal invocations
        invocations += 1
        return ToolMessage(content="ok", name=req.tool_call["name"], tool_call_id="t")

    await middleware.awrap_tool_call(
        _FakeRequest("tool", {"a": 1, "b": 2}, state=state, call_id="1"), handler
    )
    second = await middleware.awrap_tool_call(
        _FakeRequest("tool", {"b": 2, "a": 1}, state=state, call_id="2"), handler
    )

    assert invocations == 1
    assert "already made on this turn" in second.content


@pytest.mark.asyncio
async def test_new_human_message_resets_dedup_window():
    """Each new HumanMessage starts a fresh turn — prior signatures don't block."""
    middleware = DedupeToolCallsMiddleware()

    invocations = 0

    async def handler(req):
        nonlocal invocations
        invocations += 1
        return ToolMessage(content="ok", name=req.tool_call["name"], tool_call_id="t")

    state_turn1 = _state_with_messages(HumanMessage(content="first?"))
    await middleware.awrap_tool_call(
        _FakeRequest("search_docs_by_lang_chain", {"query": "x"}, state=state_turn1),
        handler,
    )

    # New turn: human follow-up message appears after an AI/tool exchange.
    state_turn2 = _state_with_messages(
        HumanMessage(content="first?"),
        AIMessage(content="answer"),
        HumanMessage(content="follow up?"),
    )
    await middleware.awrap_tool_call(
        _FakeRequest("search_docs_by_lang_chain", {"query": "x"}, state=state_turn2),
        handler,
    )

    # Both calls ran because they belong to different turns.
    assert invocations == 2
