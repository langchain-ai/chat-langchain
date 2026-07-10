"""Tests for Chat LangChain LangGraph auth hooks."""

import asyncio
from types import SimpleNamespace

import pytest

from src.api.auth import (
    _check_rate_limit,
    _configured_supabase_regions,
    _get_client_ip,
    _legacy_identity,
    _reset_rate_limit_for_tests,
    _supabase_config_for_region,
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


def test_legacy_auth_allows_polly_ids(monkeypatch):
    """Polly's temporary raw ID prefix should work during legacy migration."""
    monkeypatch.setenv("ALLOW_LEGACY_USER_ID_AUTH", "true")
    polly_id = "polly-f8cd79e3-68b6-4227-92d8-cae7488e41bf"

    assert _legacy_identity(polly_id) == polly_id


def test_legacy_auth_uses_polly_prefix(monkeypatch):
    """Polly legacy IDs should be checked like user IDs: by prefix only."""
    monkeypatch.setenv("ALLOW_LEGACY_USER_ID_AUTH", "true")

    assert _legacy_identity("polly") is None
    assert _legacy_identity("polly-not-a-uuid") == "polly-not-a-uuid"


def test_legacy_auth_rejects_polly_ids_when_disabled(monkeypatch):
    """The Polly compatibility path should turn off with legacy auth."""
    monkeypatch.setenv("ALLOW_LEGACY_USER_ID_AUTH", "false")

    assert _legacy_identity("polly-f8cd79e3-68b6-4227-92d8-cae7488e41bf") is None


def test_supabase_config_supports_regional_projects(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://us.example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "us-anon")
    monkeypatch.setenv("SUPABASE_EU_URL", "https://eu.example.supabase.co")
    monkeypatch.setenv("SUPABASE_EU_ANON_KEY", "eu-anon")
    monkeypatch.setenv("SUPABASE_APAC_URL", "https://apac.example.supabase.co")
    monkeypatch.setenv("SUPABASE_APAC_ANON_KEY", "apac-anon")
    monkeypatch.setenv("SUPABASE_AWS_URL", "https://aws.example.supabase.co")
    monkeypatch.setenv("SUPABASE_AWS_ANON_KEY", "aws-anon")

    assert _supabase_config_for_region("us") == (
        "https://us.example.supabase.co",
        "us-anon",
    )
    assert _supabase_config_for_region("eu") == (
        "https://eu.example.supabase.co",
        "eu-anon",
    )
    assert _supabase_config_for_region("apac") == (
        "https://apac.example.supabase.co",
        "apac-anon",
    )
    assert _supabase_config_for_region("aws") == (
        "https://aws.example.supabase.co",
        "aws-anon",
    )
    assert _configured_supabase_regions() == [
        ("us", "https://us.example.supabase.co", "us-anon"),
        ("eu", "https://eu.example.supabase.co", "eu-anon"),
        ("apac", "https://apac.example.supabase.co", "apac-anon"),
        ("aws", "https://aws.example.supabase.co", "aws-anon"),
    ]


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
