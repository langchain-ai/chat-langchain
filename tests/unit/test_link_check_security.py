"""Tests for link-checking destination restrictions."""

import asyncio
import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools.link_check_tools import (
    LinkCheckResult,
    _cache,
    _check_single_url,
    check_links,
)


@pytest.fixture(autouse=True)
def public_dns():
    _cache.clear()
    with patch(
        "src.tools.link_check_tools.socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
    ) as resolver:
        yield resolver
    _cache.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://example.com/",
        "https://evil-langchain.com/",
        "https://langchain.com.example.com/",
        "https://webhook.site/",
        "https://oast.fun/",
        "https://oast.site/",
        "ftp://docs.langchain.com/",
        "https://user:password@docs.langchain.com/",
        "https://docs.langchain.com:invalid/",
        "https://docs.langchain.com:99999/",
        "not-a-url",
    ],
)
async def test_disallowed_urls_make_no_requests(url, public_dns):
    client = AsyncMock(spec=httpx.AsyncClient)

    result = await _check_single_url(client, url, timeout=1.0)

    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    assert result.status_code is None
    client.head.assert_not_called()
    client.get.assert_not_called()
    client.stream.assert_not_called()
    public_dns.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "::",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
async def test_allowlisted_hosts_reject_any_nonpublic_address(address, public_dns):
    public_dns.return_value.append(
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 443),
        )
    )
    client = AsyncMock(spec=httpx.AsyncClient)

    result = await _check_single_url(client, "https://docs.langchain.com/", timeout=1.0)

    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    client.head.assert_not_called()
    client.get.assert_not_called()
    client.stream.assert_not_called()


@pytest.mark.asyncio
async def test_dns_errors_are_generic_and_make_no_requests(public_dns):
    public_dns.side_effect = socket.gaierror("sensitive DNS error")
    client = AsyncMock(spec=httpx.AsyncClient)

    result = await _check_single_url(client, "https://docs.langchain.com/", timeout=1.0)

    assert result.error == "Host not allowed for link checking"
    client.head.assert_not_called()
    client.get.assert_not_called()
    client.stream.assert_not_called()


@pytest.mark.asyncio
async def test_empty_dns_result_makes_no_requests(public_dns):
    public_dns.return_value = []
    client = AsyncMock(spec=httpx.AsyncClient)

    result = await _check_single_url(client, "https://docs.langchain.com/", timeout=1.0)

    assert result.error == "Host not allowed for link checking"
    client.head.assert_not_called()
    client.get.assert_not_called()
    client.stream.assert_not_called()


@pytest.mark.asyncio
async def test_cached_success_cannot_bypass_dns_guard(public_dns):
    url = "https://docs.langchain.com/"
    _cache[url] = LinkCheckResult(url=url, valid=True, status_code=200)
    public_dns.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
    ]
    client = AsyncMock(spec=httpx.AsyncClient)

    result = await _check_single_url(client, url, timeout=1.0)

    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    client.stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["docs.langchain.com", "github.com"])
@pytest.mark.parametrize(
    "target", ["http://169.254.169.254/latest/meta-data/", "https://example.com/"]
)
async def test_redirects_to_disallowed_hosts_are_not_requested(host, target):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": target})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        result = await _check_single_url(client, f"https://{host}/", timeout=1.0)

    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    assert len(requests) == 1
    assert requests[0].url.host == host


@pytest.mark.asyncio
async def test_redirect_dns_must_be_public(public_dns):
    public_dns.side_effect = [
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))],
    ]
    client = AsyncMock(spec=httpx.AsyncClient)
    client.head.return_value = httpx.Response(
        302,
        headers={"location": "https://docs.langchain.com/"},
        request=httpx.Request("HEAD", "https://github.com/"),
    )

    result = await _check_single_url(client, "https://github.com/", timeout=1.0)

    assert result.error == "Host not allowed for link checking"
    assert client.head.await_count == 1
    client.get.assert_not_called()
    client.stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["/new-page", None])
async def test_redirect_location_is_checked_without_following(location):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": location} if location else {})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        result = await _check_single_url(
            client, "https://docs.langchain.com/old-page", timeout=1.0
        )

    assert result.valid == bool(location)
    assert result.final_url == (
        "https://docs.langchain.com/new-page" if location else None
    )
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        "docs.langchain.com",
        "python.langchain.com",
        "js.langchain.com",
        "reference.langchain.com",
        "support.langchain.com",
        "langchain.com",
        "www.langchain.com",
        "langchain.dev",
        "github.com",
    ],
)
async def test_allowed_documentation_urls_validate(host):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text="<title>Documentation</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, f"https://{host}/", timeout=1.0)

    assert result.valid
    assert result.status_code == 200
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_head_fallback_does_not_follow_redirects():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        result = await _check_single_url(client, "https://github.com/", timeout=1.0)

    assert result.error == "Host not allowed for link checking"
    assert [request.method for request in requests] == ["HEAD", "GET"]
    assert all(request.url.host == "github.com" for request in requests)


@pytest.mark.asyncio
async def test_head_fallback_rechecks_dns(public_dns):
    public_dns.side_effect = [
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    client = AsyncMock(spec=httpx.AsyncClient)
    client.head.return_value = httpx.Response(405)

    result = await _check_single_url(client, "https://github.com/", timeout=1.0)

    assert result.error == "Host not allowed for link checking"
    assert client.head.await_count == 1
    client.get.assert_not_called()


def test_tool_rejects_internal_and_external_urls_without_requests():
    with patch("src.tools.link_check_tools.httpx.AsyncClient", autospec=True) as client:
        result = asyncio.run(
            check_links.ainvoke({"urls": ["http://127.0.0.1/", "https://example.com/"]})
        )

    assert "0/2 valid" in result
    assert result.count("Host not allowed for link checking") == 2
    client.assert_called_once_with(
        headers={"User-Agent": "LangChain-LinkChecker/1.0"},
        follow_redirects=False,
        trust_env=False,
    )
    connection = client.return_value.__aenter__.return_value
    connection.head.assert_not_called()
    connection.get.assert_not_called()
    connection.stream.assert_not_called()
