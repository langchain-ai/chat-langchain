"""Tests for Chat LangChain LangGraph auth hooks."""

import asyncio
from types import SimpleNamespace

from src.api.auth import enrich_run_metadata, update_owner_metadata


def test_thread_update_stamps_authenticated_user_id():
    """Thread metadata updates should preserve the authenticated owner invariant."""
    ctx = SimpleNamespace(user={"identity": "user@example.com"})
    value = {"metadata": {"title": "Existing title", "user_id": "spoofed"}}

    filters = asyncio.run(update_owner_metadata(ctx, value))

    assert filters == {"user_id": "user@example.com"}
    assert value["metadata"]["user_id"] == "user@example.com"
    assert value["metadata"]["title"] == "Existing title"


def test_create_run_returns_owner_filter_and_run_metadata():
    """Run creation should be scoped to the authenticated thread owner."""
    ctx = SimpleNamespace(user={"identity": "user@example.com"})
    value = {
        "assistant_id": "docs_agent",
        "kwargs": {
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "config": {},
        },
    }

    filters = asyncio.run(enrich_run_metadata(ctx, value))

    assert filters == {"user_id": "user@example.com"}
    assert value["metadata"]["user_id"] == "user@example.com"
    assert value["metadata"]["source_type"] == "Chat-LangChain"
