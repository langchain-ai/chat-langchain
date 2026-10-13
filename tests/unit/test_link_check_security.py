"""Test URL, DNS, and redirect restrictions without outbound requests."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools.link_check_tools import (
    MAX_REDIRECTS,
    LinkCheckResult,
    _cache,
    _check_single_url,
    _is_allowed_url,
    check_links,
)


@pytest.fixture(autouse=True)
def public_dns():
    _cache.clear()
    with patch("asyncio.BaseEventLoop.getaddrinfo", new_callable=AsyncMock) as dns:
        dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        yield dns
    _cache.clear()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://webhook.site/canary",
        "https://docs.langchain.com.evil.example/",
        "https://docs.langchain.com@evil.example/",
        "https://github.com/other/project",
        "ftp://docs.langchain.com/",
        "not-a-url",
    ],
)
def test_disallowed_urls_never_resolve_or_request_even_when_cached(url, public_dns):
    _cache[url] = LinkCheckResult(url=url, valid=True)
    client = AsyncMock()

    result = asyncio.run(_check_single_url(client, url, 1.0))

    assert not result.valid
    assert result.error == "Host not allowed"
    public_dns.assert_not_called()
    assert not client.mock_calls


@pytest.mark.parametrize(
    "url",
    [
        "https://DOCS.LANGCHAIN.COM:443/guide",
        "https://user:password@docs.langchain.com/guide",
        "http://reference.langchain.com/",
        "https://github.com/langchain-ai/langchain",
    ],
)
def test_allowlist_matches_hostname_not_netloc(url):
    assert _is_allowed_url(url)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff0e::1",
        "4000::1",
        "::",
        "::ffff:127.0.0.1",
    ],
)
def test_any_non_public_dns_address_blocks_request(address, public_dns):
    public_dns.return_value.append((2, 1, 6, "", (address, 443)))
    client = AsyncMock()
    url = "https://docs.langchain.com/guide"
    _cache[url] = LinkCheckResult(url=url, valid=True)

    result = asyncio.run(_check_single_url(client, url, 1.0))

    assert not result.valid
    assert result.error == "Host not allowed"
    assert not client.mock_calls


@pytest.mark.parametrize(
    "start_host", ["docs.langchain.com", "reference.langchain.com"]
)
@pytest.mark.parametrize(
    "location",
    [
        "https://webhook.site/canary",
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_redirect_to_external_host_is_blocked(start_host, location):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": location})

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ) as client:
            return await _check_single_url(client, f"https://{start_host}/guide", 1.0)

    result = asyncio.run(run())

    assert result.error == "Host not allowed"
    assert len(requests) == 1
    assert requests[0].url.host == start_host


def test_redirect_to_allowed_host_with_private_dns_is_blocked(public_dns):
    requests = []

    def handler(request):
        requests.append(request)
        public_dns.return_value = [(2, 1, 6, "", ("10.0.0.1", 443))]
        return httpx.Response(
            302, headers={"Location": "https://support.langchain.com/"}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _check_single_url(client, "https://docs.langchain.com/", 1.0)

    result = asyncio.run(run())

    assert result.error == "Host not allowed"
    assert len(requests) == 1


def test_cached_redirect_destination_is_rechecked():
    url = "https://docs.langchain.com/"
    _cache[url] = LinkCheckResult(
        url=url, valid=True, final_url="https://webhook.site/canary"
    )
    client = AsyncMock()

    result = asyncio.run(_check_single_url(client, url, 1.0))

    assert result.error == "Host not allowed"
    assert not client.mock_calls


@pytest.mark.parametrize("location", ["/next", "https://docs.langchain.com/next"])
def test_allowed_redirect_preserves_soft_404_detection(location):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": location})
        return httpx.Response(200, text="<title>Page not found</title>")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _check_single_url(client, "https://docs.langchain.com/", 1.0)

    result = asyncio.run(run())

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert requests[-1].method == "GET"


@pytest.mark.parametrize("private_fallback", [False, True])
def test_head_to_get_fallback_is_guarded(private_fallback, public_dns):
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "HEAD":
            if private_fallback:
                public_dns.return_value = [(2, 1, 6, "", ("10.0.0.1", 443))]
            return httpx.Response(405)
        return httpx.Response(302, headers={"Location": "http://127.0.0.1:8080/"})

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ) as client:
            return await _check_single_url(
                client, "https://reference.langchain.com/", 1.0
            )

    result = asyncio.run(run())

    assert result.error == "Host not allowed"
    assert [request.method for request in requests] == (
        ["HEAD"] if private_fallback else ["HEAD", "GET"]
    )


def test_redirect_count_is_bounded():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": f"/hop/{len(requests)}"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _check_single_url(client, "https://docs.langchain.com/", 1.0)

    result = asyncio.run(run())

    assert result.error == "Too many redirects"
    assert len(requests) == MAX_REDIRECTS + 1


def test_normal_docs_url_is_valid():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text="Documentation")
            )
        ) as client:
            return await _check_single_url(
                client, "https://docs.langchain.com/guide", 1.0
            )

    result = asyncio.run(run())

    assert result.valid
    assert result.status_code == 200


def test_dns_errors_are_generic(public_dns):
    public_dns.side_effect = OSError("secret internal DNS details")
    result = asyncio.run(
        _check_single_url(AsyncMock(), "https://docs.langchain.com/", 1.0)
    )
    assert result.error == "Unreachable"


def test_connection_errors_are_generic():
    client = AsyncMock()
    client.head.side_effect = httpx.ConnectError("secret internal connection details")
    result = asyncio.run(
        _check_single_url(client, "https://reference.langchain.com/", 1.0)
    )
    assert result.error == "Unreachable"


def test_tool_call_on_unapproved_url_makes_no_request(public_dns):
    with patch("src.tools.link_check_tools.httpx.AsyncClient") as client:
        result = asyncio.run(
            check_links.ainvoke({"urls": ["https://webhook.site/canary"]})
        )
        assert not client.return_value.__aenter__.return_value.mock_calls

    assert "Host not allowed" in result
    public_dns.assert_not_called()
