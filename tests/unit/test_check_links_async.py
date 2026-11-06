"""Tests for the check_links tool — verifies it works in both sync and async contexts.

The core bug: `asyncio.run()` raises `RuntimeError: This event loop is already running`
when called from within an async context (e.g. a LangGraph agent node). The fix is to
make `check_links` itself `async` so it can simply `await _check_urls_async(...)`.

Test strategy: use `unittest.mock` to patch the internal HTTP layer so the tests are
fast, deterministic, and require no real network access or LangSmith credentials.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake LinkCheckResult objects without importing the whole
# module (which would trigger import-time side effects).
# ---------------------------------------------------------------------------
from src.tools.link_check_tools import (
    LinkCheckResult,
    _check_single_url,
    _check_urls_async,
    check_links,
)


class _FakeStreamResponse:
    """Minimal async streaming response for _check_single_url tests."""

    def __init__(self, url: str, status_code: int, content: str):
        self.url = url
        self.status_code = status_code
        self._content = content

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return None

    async def aiter_text(self):
        yield self._content


class _FakeStreamingClient:
    """Minimal client that exercises the soft-404 streaming path."""

    def __init__(self, content: str, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def stream(self, method: str, url: str, **kwargs):  # noqa: ARG002
        return _FakeStreamResponse(url, self.status_code, self.content)


# ---------------------------------------------------------------------------
# Fixture: a canned async replacement for _check_urls_async
# ---------------------------------------------------------------------------


def _make_async_check_mock(results: list[LinkCheckResult]):
    """Return an async function that ignores its arguments and returns *results*."""

    async def _mock_check_urls_async(urls, timeout):  # noqa: ARG001
        return results

    return _mock_check_urls_async


# ===========================================================================
# 1. Test that check_links works when called from within an async context
#    (this is the regression test — it FAILS before the fix because the old
#    code calls asyncio.run() inside an already-running event loop).
# ===========================================================================


@pytest.mark.asyncio
async def test_check_links_works_in_async_context():
    """check_links must be awaitable and must NOT raise RuntimeError in async context.

    Before the fix, calling `check_links.invoke(...)` from inside a coroutine would
    ultimately hit `asyncio.run()` which raises:
        RuntimeError: This event loop is already running
    After the fix, check_links is `async def` so it can be awaited directly.
    """
    fake_results = [
        LinkCheckResult(url="https://example.com", valid=True, status_code=200),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        # After the fix check_links is a coroutine function (async def) wrapped
        # by @tool.  LangChain's StructuredTool exposes it as .acall / .ainvoke.
        # We call the underlying coroutine function directly to keep the test
        # independent of LangChain's tool-wrapping internals.
        result = await check_links.acall({"urls": ["https://example.com"]})

    assert "1/1 valid" in result
    assert "  - https://example.com" in result


@pytest.mark.asyncio
async def test_check_links_async_context_does_not_raise_runtime_error():
    """Explicitly assert that no RuntimeError is raised in an async context."""
    fake_results = [
        LinkCheckResult(url="https://docs.langchain.com/", valid=True, status_code=200),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        raised = False
        try:
            await check_links.acall({"urls": ["https://docs.langchain.com/"]})
        except RuntimeError as exc:
            if "event loop is already running" in str(exc):
                raised = True

    assert not raised, (
        "RuntimeError('This event loop is already running') was raised — "
        "the fix has not been applied correctly."
    )


# ===========================================================================
# 2. Test that check_links works normally in a sync context
#    (the tool must still be usable when there is no running event loop,
#     e.g. in plain scripts or sync test runners).
# ===========================================================================


def test_check_links_works_in_sync_context():
    """check_links.invoke() must work when called outside an async context."""
    fake_results = [
        LinkCheckResult(url="https://example.com", valid=True, status_code=200),
        LinkCheckResult(
            url="https://bad.example.com",
            valid=False,
            status_code=404,
            error="HTTP 404",
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = check_links.invoke(
            {"urls": ["https://example.com", "https://bad.example.com"]}
        )

    assert "1/2 valid" in result
    assert "  - https://example.com" in result
    assert "  - https://bad.example.com:" in result
    assert "HTTP 404" in result


def test_check_links_sync_deduplicates_urls():
    """Duplicate URLs should only be checked once."""
    seen_urls: list[list[str]] = []

    async def _recording_mock(urls, timeout):  # noqa: ARG001
        seen_urls.append(list(urls))
        return [LinkCheckResult(url=u, valid=True, status_code=200) for u in urls]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_recording_mock,
    ):
        result = check_links.invoke(
            {
                "urls": [
                    "https://example.com",
                    "https://example.com",
                    "https://example.com",
                ]
            }
        )

    # _check_urls_async should have been called with exactly ONE unique URL
    assert len(seen_urls) == 1
    assert seen_urls[0] == ["https://example.com"]
    assert "1/1 valid" in result


# ===========================================================================
# 3. Edge cases
# ===========================================================================


def test_check_links_empty_list():
    """Passing an empty list should return the 'no URLs' message without error."""
    # No mock needed — the early-return branch is hit before any async work.
    result = check_links.invoke({"urls": []})
    assert result == "No URLs provided to check."


@pytest.mark.asyncio
async def test_check_links_async_empty_list():
    """Empty list in async context should also return the 'no URLs' message."""
    result = await check_links.acall({"urls": []})
    assert result == "No URLs provided to check."


def test_check_links_invalid_url_format():
    """An invalid URL (not http/https) must be reported as invalid."""
    # We do NOT mock here — _check_single_url validates format synchronously via
    # _is_valid_url before making any network calls, so no HTTP is performed.
    # We still mock _check_urls_async to avoid needing a real event loop in the
    # sync path and to keep the test fast.
    fake_results = [
        LinkCheckResult(url="not-a-url", valid=False, error="Invalid URL format"),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = check_links.invoke({"urls": ["not-a-url"]})

    assert "0/1 valid" in result
    assert "Invalid URL format" in result


@pytest.mark.asyncio
async def test_check_links_async_invalid_url_format():
    """Invalid URL in async context should also be reported correctly."""
    fake_results = [
        LinkCheckResult(
            url="ftp://not-supported", valid=False, error="Invalid URL format"
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = await check_links.acall({"urls": ["ftp://not-supported"]})

    assert "0/1 valid" in result
    assert "Invalid URL format" in result


@pytest.mark.asyncio
async def test_support_article_not_found_is_invalid_soft_404():
    """Support articles can return HTTP 200 while rendering an article-not-found page."""
    result = await _check_single_url(
        _FakeStreamingClient("<html><body><div>Article Not Found</div></body></html>"),
        "https://support.langchain.com/hc/en-us/articles/missing-article",
        timeout=1.0,
    )

    assert not result.valid
    assert result.status_code == 200
    assert result.error == "Soft 404: Page shows 'not found' content"


@pytest.mark.asyncio
async def test_support_article_normal_content_is_valid():
    """Normal support article content should still be treated as valid."""
    result = await _check_single_url(
        _FakeStreamingClient(
            "<html><body><div>Useful support article</div></body></html>"
        ),
        "https://support.langchain.com/hc/en-us/articles/existing-article",
        timeout=1.0,
    )

    assert result.valid
    assert result.status_code == 200
    assert result.error is None


@pytest.fixture(autouse=True)
def public_dns():
    from src.tools import link_check_tools

    link_check_tools._cache.clear()
    with patch(
        "socket.getaddrinfo",
        return_value=[
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ],
    ) as resolver:
        with patch(
            "httpx.AsyncHTTPTransport.handle_async_request",
            side_effect=AssertionError("Unexpected network request"),
        ):
            yield resolver
    link_check_tools._cache.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://224.0.0.1/",
        "http://127.0.0.1:400/",
        "https://user:password@example.com/",
    ],
)
async def test_non_public_targets_never_connect(url, public_dns):
    from src.tools import link_check_tools

    public_dns.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]
    link_check_tools._cache[url] = LinkCheckResult(url=url, valid=True)
    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        new_callable=AsyncMock,
    ) as send:
        result = await _check_urls_async([url], 1.0)
    assert not result[0].valid
    assert "rejected" in result[0].error
    send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["10.0.0.1", "fc00::1"])
async def test_mixed_dns_answers_never_connect(public_dns, address):
    from src.tools import link_check_tools

    public_dns.return_value.append((2, 1, 6, "", (address, 443)))
    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        new_callable=AsyncMock,
    ) as send:
        result = await _check_urls_async(["https://docs.langchain.com/"], 1.0)
    assert not result[0].valid
    send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", ["http://127.0.0.1/", "http://[::1]/", "http://internal.example/"]
)
async def test_private_redirect_never_connects(target, public_dns):
    import httpx

    from src.tools import link_check_tools

    public_dns.side_effect = lambda host, *args, **kwargs: [
        (
            2,
            1,
            6,
            "",
            ("10.0.0.1" if host == "internal.example" else "93.184.216.34", 443),
        )
    ]
    requests = []

    async def send(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": target})

    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        side_effect=send,
    ):
        results = await _check_urls_async(["https://docs.langchain.com/start"], 1.0)
    assert len(requests) == 1
    assert not results[0].valid
    assert "rejected" in results[0].error


@pytest.mark.asyncio
async def test_public_documentation_pins_ip_and_preserves_hostname(public_dns):
    import httpx

    from src.tools import link_check_tools

    requests = []

    async def send(request):
        requests.append(request)
        return httpx.Response(200, text="Useful documentation")

    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        side_effect=send,
    ):
        results = await _check_urls_async(["https://docs.langchain.com/guide"], 1.0)
    assert results[0].valid
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "docs.langchain.com"
    assert requests[0].extensions["sni_hostname"] == "docs.langchain.com"


@pytest.mark.asyncio
async def test_dns_rebinding_never_connects(public_dns):
    from src.tools import link_check_tools

    public_dns.side_effect = [
        [(2, 1, 6, "", ("93.184.216.34", 443))],
        [(2, 1, 6, "", ("127.0.0.1", 443))],
    ]
    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        new_callable=AsyncMock,
    ) as send:
        results = await _check_urls_async(["https://docs.langchain.com/"], 1.0)
    assert not results[0].valid
    send.assert_not_called()


@pytest.mark.asyncio
async def test_oversized_batch_never_resolves_or_connects(public_dns):
    from src.tools import link_check_tools

    urls = ["https://docs.langchain.com/"] * (link_check_tools.MAX_URLS + 1)
    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        new_callable=AsyncMock,
    ) as send:
        results = await _check_urls_async(urls, 1.0)
        output = await check_links.ainvoke({"urls": urls})
    assert all(
        not result.valid and "Batch exceeds" in result.error for result in results
    )
    assert "no URLs checked" in output
    public_dns.assert_not_called()
    send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, 11, float("inf"), float("nan")])
async def test_invalid_timeout_never_resolves(timeout, public_dns):
    results = await _check_urls_async(["https://docs.langchain.com/"], timeout)
    assert not results[0].valid
    assert "Timeout must be" in results[0].error
    public_dns.assert_not_called()


@pytest.mark.asyncio
async def test_concurrency_is_bounded():
    import httpx

    from src.tools import link_check_tools

    active = 0
    peak = 0

    async def send(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, text="Documentation")

    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        side_effect=send,
    ):
        results = await _check_urls_async(
            [f"https://docs.langchain.com/{i}" for i in range(15)], 1.0
        )
    assert all(result.valid for result in results)
    assert peak <= link_check_tools.MAX_CONCURRENCY


@pytest.mark.asyncio
async def test_total_timeout_includes_dns(public_dns):
    import time

    def slow_dns(*args, **kwargs):
        time.sleep(0.05)
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    public_dns.side_effect = slow_dns
    result = await _check_urls_async(["https://docs.langchain.com/"], 0.01)
    assert not result[0].valid
    assert result[0].error == "Request timed out"


@pytest.mark.asyncio
async def test_public_redirect_and_head_fallback_remain_functional():
    import httpx

    from src.tools import link_check_tools

    requests = []

    async def send(request):
        requests.append((request.method, request.headers["host"], request.url.path))
        if request.url.path == "/start":
            return httpx.Response(
                302, headers={"Location": "https://example.com/guide"}
            )
        if request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(200, text="Public documentation")

    with patch.object(
        link_check_tools.httpx.AsyncHTTPTransport,
        "handle_async_request",
        side_effect=send,
    ):
        results = await _check_urls_async(["https://example.org/start"], 1.0)
    assert results[0].valid
    assert results[0].final_url == "https://example.com/guide"
    assert ("HEAD", "example.com", "/guide") in requests
    assert ("GET", "example.com", "/guide") in requests
