"""Test SSRF defenses without making network or DNS requests."""

import asyncio
import socket
import ssl
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools import link_check_tools as links


@pytest.fixture(autouse=True)
def clear_cache():
    links._cache.clear()
    yield
    links._cache.clear()


def _records(*addresses):
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, 443),
        )
        for address in addresses
    ]


def _check(url, handler=None, records=None, timeout=10.0, dns_error=None):
    requests = []
    resolver = AsyncMock(
        return_value=_records("93.184.216.34") if records is None else records,
        side_effect=dns_error,
    )

    async def request_handler(transport, request):
        requests.append(request)
        if handler:
            return handler(request)
        return httpx.Response(200, text="<title>Documentation</title>")

    async def run():
        with (
            patch.object(asyncio.get_running_loop(), "getaddrinfo", resolver),
            patch.object(
                httpx.AsyncHTTPTransport, "handle_async_request", request_handler
            ),
        ):
            return (await links._check_urls_async([url], timeout))[0]

    return asyncio.run(run()), requests, resolver


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://93.184.216.34/",
        "https://probe.oast.example/",
        "https://webhook.site/attacker",
        "https://langchain.com.attacker.example/",
        "https://notlangchain.com/",
    ],
)
def test_disallowed_hosts_never_resolve_or_connect(url):
    result, requests, resolver = _check(url)
    assert not result.valid
    assert result.error == "Host not allowed"
    assert not requests
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "url", ["not-a-url", "ftp://docs.langchain.com/", "https:///missing"]
)
def test_invalid_url_never_resolves_or_connects(url):
    result, requests, resolver = _check(url)
    assert result.error == "Invalid URL format"
    assert not requests
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::",
        "::ffff:127.0.0.1",
    ],
)
def test_any_nonpublic_dns_address_blocks_connection(address):
    result, requests, _ = _check(
        "https://docs.langchain.com/", records=_records("93.184.216.34", address)
    )
    assert not result.valid
    assert result.error == "unreachable"
    assert not requests


@pytest.mark.parametrize(
    "records,dns_error",
    [
        ([], None),
        (None, socket.gaierror("sensitive DNS failure")),
    ],
)
def test_dns_fails_closed(records, dns_error):
    result, requests, _ = _check(
        "https://docs.langchain.com/", records=records, dns_error=dns_error
    )
    assert result.error == "unreachable"
    assert not requests


@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "https://probe.oast.example/",
        "ftp://docs.langchain.com/",
    ],
)
def test_redirect_target_is_checked_before_request(location):
    result, requests, resolver = _check(
        "https://docs.langchain.com/",
        lambda request: httpx.Response(302, headers={"Location": location}),
    )
    assert not result.valid
    assert result.error in ("Host not allowed", "Invalid URL format")
    assert len(requests) == 1
    assert resolver.await_count == 1


def test_redirect_to_allowed_host_with_private_dns_is_blocked():
    resolver = AsyncMock(side_effect=[_records("93.184.216.34"), _records("10.0.0.1")])
    network = AsyncMock(
        return_value=httpx.Response(
            302, headers={"Location": "https://support.langchain.com/article"}
        )
    )

    async def run():
        with (
            patch.object(asyncio.get_running_loop(), "getaddrinfo", resolver),
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", network),
        ):
            return await links._check_urls_async(["https://docs.langchain.com/"], 10)

    result = asyncio.run(run())[0]
    assert result.error == "unreachable"
    assert network.await_count == 1
    assert resolver.await_count == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/oss/python/langchain/overview",
        "https://DOCS.LANGCHAIN.COM./",
        "https://new.docs.langchain.com/",
        "http://docs.langchain.com:8080/",
        "https://github.com/langchain-ai/langchain",
        "https://pypi.org/project/langchain/",
        "https://www.npmjs.com/package/langchain",
    ],
)
def test_allowed_urls_pin_address_and_preserve_hostname(url):
    result, requests, resolver = _check(url)
    assert result.valid
    assert len(requests) == 1
    request = requests[0]
    original = httpx.URL(url)
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == original.netloc.decode("ascii")
    assert request.extensions["sni_hostname"] == original.raw_host.decode("ascii")
    resolver.assert_awaited_once_with(
        original.raw_host.decode("ascii"),
        original.port or (443 if original.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )


def test_relative_redirect_revalidates_dns():
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(302, headers={"Location": "/new"})
        return httpx.Response(200, text="<title>Documentation</title>")

    result, requests, resolver = _check("https://docs.langchain.com/old", handler)
    assert result.valid
    assert result.final_url == "https://docs.langchain.com/new"
    assert len(requests) == 2
    assert resolver.await_count == 2


def test_redirect_limit():
    result, requests, _ = _check(
        "https://docs.langchain.com/",
        lambda request: httpx.Response(302, headers={"Location": "/next"}),
    )
    assert result.error == "Too many redirects"
    assert len(requests) == links.MAX_REDIRECTS + 1


def test_head_to_get_fallback_revalidates_dns():
    def handler(request):
        return httpx.Response(405 if request.method == "HEAD" else 200)

    result, requests, resolver = _check(
        "https://github.com/langchain-ai/langchain", handler
    )
    assert result.valid
    assert [request.method for request in requests] == ["HEAD", "GET"]
    assert resolver.await_count == 2


def test_fallback_blocks_changed_dns():
    resolver = AsyncMock(side_effect=[_records("93.184.216.34"), _records("127.0.0.1")])
    network = AsyncMock(return_value=httpx.Response(405))

    async def run():
        with (
            patch.object(asyncio.get_running_loop(), "getaddrinfo", resolver),
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", network),
        ):
            return await links._check_urls_async(
                ["https://github.com/langchain-ai/langchain"], 10
            )

    assert asyncio.run(run())[0].error == "unreachable"
    assert network.await_count == 1


@pytest.mark.parametrize(
    "content", ["<title>404 Not Found</title>", "Article Not Found"]
)
def test_soft_404_still_detected(content):
    result, _, _ = _check(
        "https://docs.langchain.com/missing",
        lambda request: httpx.Response(200, text=content),
    )
    assert not result.valid
    assert result.status_code == 200
    assert result.error == "Soft 404: Page shows 'not found' content"


@pytest.mark.parametrize(
    "timeout,expected",
    [
        (-20, 1),
        (0, 1),
        (5, 5),
        (100, 10),
        (float("inf"), 10),
        (float("nan"), 10),
    ],
)
def test_timeout_is_clamped(timeout, expected):
    _, requests, _ = _check("https://docs.langchain.com/", timeout=timeout)
    assert requests[0].extensions["timeout"]["connect"] == expected


def test_connection_errors_are_sanitized():
    def handler(request):
        raise httpx.ConnectError("secret internal IP and network details")

    result, _, _ = _check("https://docs.langchain.com/", handler)
    assert result.error == "unreachable"


def test_tool_async_interface_deduplicates_and_clamps_timeout():
    url = "https://docs.langchain.com/"
    checker = AsyncMock(return_value=[links.LinkCheckResult(url=url, valid=True)])
    with patch.object(links, "_check_urls_async", checker):
        result = asyncio.run(
            links.check_links.ainvoke({"urls": [url, url], "timeout": 1000})
        )
    checker.assert_awaited_once_with([url], 10)
    assert "1/1 valid" in result


def test_tool_async_interface_accepts_empty_list():
    assert (
        asyncio.run(links.check_links.ainvoke({"urls": []}))
        == "No URLs provided to check."
    )


def test_cache_is_bounded():
    for index in range(links.MAX_CACHE_SIZE + 20):
        links._cache_result(
            links.LinkCheckResult(
                url=f"https://probe.oast.example/{index}", valid=False
            )
        )
    assert len(links._cache) == links.MAX_CACHE_SIZE
    assert "https://probe.oast.example/0" not in links._cache


def test_underlying_connection_uses_pinned_ip_and_original_tls_hostname():
    stream = AsyncMock()
    stream.start_tls.return_value = stream
    stream.read.return_value = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
    stream.get_extra_info = lambda name: None
    backend = AsyncMock()
    backend.connect_tcp.return_value = stream
    resolver = AsyncMock(side_effect=[_records("93.184.216.34"), _records("127.0.0.1")])

    async def run():
        async with links._SafeTransport() as transport:
            transport._transport._pool._network_backend = backend
            with patch.object(asyncio.get_running_loop(), "getaddrinfo", resolver):
                async with httpx.AsyncClient(
                    transport=transport, trust_env=False
                ) as client:
                    return await links._check_single_url(
                        client, "https://docs.langchain.com/", 10
                    )

    assert asyncio.run(run()).valid
    resolver.assert_awaited_once()
    assert backend.connect_tcp.await_args.kwargs["host"] == "93.184.216.34"
    tls_args = stream.start_tls.await_args.kwargs
    assert tls_args["server_hostname"] == "docs.langchain.com"
    assert tls_args["ssl_context"].check_hostname
    assert tls_args["ssl_context"].verify_mode == ssl.CERT_REQUIRED
    assert b"Host: docs.langchain.com\r\n" in stream.write.await_args_list[0].args[0]
