"""Tests for the check_links tool — verifies it works in both sync and async contexts.

The core bug: `asyncio.run()` raises `RuntimeError: This event loop is already running`
when called from within an async context (e.g. a LangGraph agent node). The fix is to
make `check_links` itself `async` so it can simply `await _check_urls_async(...)`.

Test strategy: use `unittest.mock` to patch the internal HTTP layer so the tests are
fast, deterministic, and require no real network access or LangSmith credentials.
"""

import asyncio
import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake LinkCheckResult objects without importing the whole
# module (which would trigger import-time side effects).
# ---------------------------------------------------------------------------
from src.tools.link_check_tools import (
    MAX_REDIRECTS,
    LinkCheckResult,
    _cache,
    _check_single_url,
    _check_urls_async,
    check_links,
)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    _cache.clear()
    resolver = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
    )
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolver)
    yield resolver
    _cache.clear()


@pytest.fixture
def network(monkeypatch):
    requests = []
    responses = []

    async def handle_request(transport, request):
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", handle_request
    )
    return requests, responses


class _FakeStreamResponse:
    """Minimal async streaming response for _check_single_url tests."""

    def __init__(self, url: str, status_code: int, content: str):
        self.url = url
        self.status_code = status_code
        self.headers = {}
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://loki:8000/ok",
        "http://169.254.169.254/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://93.184.216.34/",
    ],
)
async def test_reject_direct_internal_and_ip_hosts(url, network, public_dns):
    requests, _ = network

    (result,) = await _check_urls_async([url], timeout=1.0)

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert not requests
    public_dns.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "192.0.2.1",
        "240.0.0.1",
        "224.0.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
async def test_reject_any_nonpublic_dns_answer(address, network, public_dns):
    requests, _ = network
    public_dns.return_value.append(
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 443),
        )
    )
    url = "https://docs.langchain.com/oss/python/langchain/overview"

    (result,) = await _check_urls_async([url], timeout=1.0)

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("answers", [[], OSError("private DNS details")])
async def test_resolution_failure_is_private_and_not_cached(
    answers, network, public_dns
):
    requests, responses = network
    public_dns.side_effect = [answers, public_dns.return_value]
    url = "https://docs.langchain.com/overview"

    (result,) = await _check_urls_async([url], timeout=1.0)
    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert not requests

    responses.append(httpx.Response(200, text="Documentation"))
    (retried,) = await _check_urls_async([url], timeout=1.0)
    assert retried.valid


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,fallback",
    [
        ("https://docs.langchain.com/overview", False),
        ("https://www.langchain.com/pricing", False),
        ("https://www.langchain.com/pricing", True),
    ],
)
@pytest.mark.parametrize(
    "target",
    ["http://loki:8000/ok", "http://169.254.169.254/", "https://private.example/"],
)
async def test_redirect_never_requests_blocked_target(
    url, fallback, target, network, public_dns
):
    requests, responses = network
    if fallback:
        responses.append(httpx.Response(405))
    responses.append(httpx.Response(302, headers={"Location": target}))
    if "private.example" in target:
        public_dns.side_effect = [public_dns.return_value] * (2 if fallback else 1) + [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]
        ]

    (result,) = await _check_urls_async([url], timeout=1.0)

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert len(requests) == (2 if fallback else 1)
    assert all(request.url.host == "93.184.216.34" for request in requests)
    assert all(request.headers["host"] == httpx.URL(url).host for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,method",
    [
        ("https://docs.langchain.com/overview", "GET"),
        ("https://www.langchain.com/pricing", "HEAD"),
    ],
)
async def test_public_urls_preserve_hostname_and_tls_identity(url, method, network):
    requests, responses = network
    responses.append(httpx.Response(200, text="Documentation"))

    (result,) = await _check_urls_async([url], timeout=1.0)

    assert result.valid
    assert result.url == url
    assert result.final_url is None
    assert requests[0].method == method
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == httpx.URL(url).host
    assert requests[0].extensions["sni_hostname"] == httpx.URL(url).host


@pytest.mark.asyncio
@pytest.mark.parametrize("soft_404", [False, True])
async def test_relative_redirect_preserves_final_url_and_soft_404(soft_404, network):
    requests, responses = network
    responses.extend(
        [
            httpx.Response(302, headers={"Location": "/new"}),
            httpx.Response(
                200, text="<title>404 Not Found</title>" if soft_404 else "Docs"
            ),
        ]
    )

    (result,) = await _check_urls_async(["https://docs.langchain.com/old"], timeout=1.0)

    assert result.valid is not soft_404
    assert result.final_url == "https://docs.langchain.com/new"
    assert result.status_code == 200
    assert [request.url.path for request in requests] == ["/old", "/new"]


@pytest.mark.asyncio
async def test_redirect_limit(network):
    requests, responses = network
    responses.extend(
        httpx.Response(302, headers={"Location": f"/hop/{index}"})
        for index in range(MAX_REDIRECTS + 1)
    )

    (result,) = await _check_urls_async(
        ["https://docs.langchain.com/start"], timeout=1.0
    )

    assert result.error == "Too many redirects"
    assert len(requests) == MAX_REDIRECTS + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect", [False, True])
async def test_dns_rebinding_is_rejected_before_next_request(
    redirect, network, public_dns
):
    requests, responses = network
    responses.append(
        httpx.Response(302, headers={"Location": "/next"})
        if redirect
        else httpx.Response(405)
    )
    public_dns.side_effect = [
        public_dns.return_value,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    url = "https://www.langchain.com/pricing"

    (result,) = await _check_urls_async([url], timeout=1.0)

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"


@pytest.mark.asyncio
async def test_public_head_falls_back_to_get(network):
    requests, responses = network
    responses.extend([httpx.Response(405), httpx.Response(200)])

    (result,) = await _check_urls_async(
        ["https://www.langchain.com/pricing"], timeout=1.0
    )

    assert result.valid
    assert [request.method for request in requests] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_redirect_to_docs_checks_soft_404(network):
    requests, responses = network
    responses.extend(
        [
            httpx.Response(
                302, headers={"Location": "https://docs.langchain.com/missing"}
            ),
            httpx.Response(200, text="<title>Not Found</title>"),
        ]
    )

    (result,) = await _check_urls_async(["https://www.langchain.com/docs"], timeout=1.0)

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert result.final_url == "https://docs.langchain.com/missing"
    assert [request.method for request in requests] == ["HEAD", "GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["ftp://public.example/path", "file:///etc/passwd"])
async def test_redirect_rejects_non_http_schemes(target, network):
    requests, responses = network
    responses.append(httpx.Response(302, headers={"Location": target}))

    (result,) = await _check_urls_async(
        ["https://docs.langchain.com/start"], timeout=1.0
    )

    assert result.error == "Host not allowed"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,error",
    [
        (httpx.ConnectError("private connection details"), "Connection failed"),
        (httpx.ReadTimeout("private timeout details"), "Request timed out"),
        (RuntimeError("private details"), "Request failed"),
        (httpx.Response(503), "HTTP 503"),
    ],
)
async def test_transient_errors_are_generic_and_not_cached(failure, error, network):
    _, responses = network
    responses.extend([failure, httpx.Response(200)])
    url = "https://www.langchain.com/pricing"

    (result,) = await _check_urls_async([url], timeout=1.0)
    assert not result.valid
    assert result.error == error

    (retried,) = await _check_urls_async([url], timeout=1.0)
    assert retried.valid
