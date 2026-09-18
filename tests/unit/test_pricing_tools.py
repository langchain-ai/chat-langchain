"""Tests for the pricing tool's scope and fallback behavior."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from src.tools.pricing_tools import (
    PRICING_SCOPE_HEADER,
    fetch_langchain_pricing,
)


def test_fetch_pricing_prepends_scope_header_to_fresh_fetch():
    with (
        patch("src.tools.pricing_tools._cached_text", None),
        patch("src.tools.pricing_tools._cached_at", 0.0),
        patch(
            "src.tools.pricing_tools._fetch_pricing_uncached",
            new=AsyncMock(return_value="fresh pricing"),
        ),
    ):
        result = asyncio.run(fetch_langchain_pricing.coroutine())

    assert result.startswith(f"{PRICING_SCOPE_HEADER}\n\n")
    assert result.endswith("fresh pricing")


def test_fetch_pricing_prepends_scope_header_to_stale_cache():
    response = httpx.Response(
        503,
        request=httpx.Request("GET", "https://www.langchain.com/pricing"),
    )
    with (
        patch("src.tools.pricing_tools._cached_text", "stale pricing"),
        patch("src.tools.pricing_tools._cached_at", 0.0),
        patch(
            "src.tools.pricing_tools._fetch_pricing_uncached",
            new=AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "unavailable", request=response.request, response=response
                )
            ),
        ),
    ):
        result = asyncio.run(fetch_langchain_pricing.coroutine())

    assert result.startswith(f"{PRICING_SCOPE_HEADER}\n\n")
    assert result.endswith("stale pricing")


def test_fetch_pricing_prepends_scope_header_to_http_error():
    response = httpx.Response(
        503,
        request=httpx.Request("GET", "https://www.langchain.com/pricing"),
    )
    with (
        patch("src.tools.pricing_tools._cached_text", None),
        patch("src.tools.pricing_tools._cached_at", 0.0),
        patch(
            "src.tools.pricing_tools._fetch_pricing_uncached",
            new=AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "unavailable", request=response.request, response=response
                )
            ),
        ),
    ):
        result = asyncio.run(fetch_langchain_pricing.coroutine())

    assert result.startswith(f"{PRICING_SCOPE_HEADER}\n\n")
    assert "HTTP 503" in result


def test_fetch_pricing_docstring_does_not_forbid_docs_search():
    assert "DO NOT use docs search" not in fetch_langchain_pricing.description
