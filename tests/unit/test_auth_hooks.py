"""Tests for Chat LangChain LangGraph auth hooks."""

import asyncio
from types import SimpleNamespace

import pytest

from src.api.auth import (
    _check_rate_limit,
    _get_client_ip,
    _reset_rate_limit_for_tests,
    enrich_run_metadata,
    update_owner_metadata,
)


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


def test_backend_rate_limit_blocks_twenty_first_request_for_same_ip():
    """The backend should cap each IP at 20 requests per minute."""
    _reset_rate_limit_for_tests()
    headers = {"x-forwarded-for": "1.2.3.4"}

    for i in range(20):
        _check_rate_limit(headers, now=1000 + i)

    with pytest.raises(Exception) as exc_info:
        _check_rate_limit(headers, now=1020)

    assert getattr(exc_info.value, "status_code", None) == 429


def test_backend_rate_limit_uses_independent_ip_buckets():
    """A blocked IP should not consume another IP's request bucket."""
    _reset_rate_limit_for_tests()

    for i in range(20):
        _check_rate_limit({"x-forwarded-for": "1.2.3.4"}, now=1000 + i)

    _check_rate_limit({"x-forwarded-for": "5.6.7.8"}, now=1020)


def test_backend_rate_limit_extracts_ip_from_byte_headers():
    """LangGraph may inject request headers as dict[bytes, bytes]."""
    headers = {b"x-forwarded-for": b"1.2.3.4, 5.6.7.8"}

    assert _get_client_ip(headers) == "1.2.3.4"


def test_backend_rate_limit_uses_independent_byte_header_ip_buckets():
    """Byte header keys should not collapse all callers into the unknown bucket."""
    _reset_rate_limit_for_tests()

    for i in range(20):
        _check_rate_limit({b"x-forwarded-for": b"1.2.3.4"}, now=1000 + i)

    _check_rate_limit({b"x-forwarded-for": b"5.6.7.8"}, now=1020)


def test_backend_rate_limit_window_expires_old_requests():
    """Requests should be allowed again once they fall out of the window."""
    _reset_rate_limit_for_tests()
    headers = {"x-forwarded-for": "1.2.3.4"}

    for i in range(20):
        _check_rate_limit(headers, now=1000 + i)

    _check_rate_limit(headers, now=1061)


def test_create_run_rate_limit_blocks_twenty_first_request_for_same_ip():
    """Rate limiting should apply to run creation, not every auth check."""
    _reset_rate_limit_for_tests()
    ctx = SimpleNamespace(user={"identity": "user@example.com", "client_ip": "1.2.3.4"})

    for _ in range(20):
        value = {
            "assistant_id": "docs_agent",
            "kwargs": {
                "input": {"messages": [{"role": "user", "content": "hello"}]},
                "config": {},
            },
        }
        asyncio.run(enrich_run_metadata(ctx, value))

    value = {
        "assistant_id": "docs_agent",
        "kwargs": {
            "input": {"messages": [{"role": "user", "content": "hello"}]},
            "config": {},
        },
    }
    with pytest.raises(Exception) as exc_info:
        asyncio.run(enrich_run_metadata(ctx, value))

    assert getattr(exc_info.value, "status_code", None) == 429
