"""Test citation URL restrictions and connection-bound address validation."""

import asyncio
import socket
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.tools import link_check_tools as links

PUBLIC_IP = "93.184.216.34"


def _records(*addresses):
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 443),
        )
        for address in addresses
    ]


@pytest.fixture(autouse=True)
def clean_cache():
    links._cache.clear()
    yield
    links._cache.clear()


@pytest.fixture
def dns(monkeypatch):
    resolver = AsyncMock(return_value=_records(PUBLIC_IP))
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolver)
    return resolver


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/",
        "https://example.com/",
        "https://evillangchain.com/",
        "https://docs.langchain.com.evil.test/",
        "https://github.com/other/repo",
        "https://github.com/langchain-ai",
        "https://github.com/langchain-ai/../other/repo",
        "https://github.com/langchain-ai/%2e%2e/other",
        "https://github.com/langchain-ai%2fother/repo",
        "https://user:password@docs.langchain.com/",
        "https://user@docs.langchain.com/",
        "http://[::1]/",
        "ftp://docs.langchain.com/",
        "https://docs.langchain.com\\@example.com/",
        "https://docs.langchain.com/\n",
        "not-a-url",
    ],
)
@pytest.mark.asyncio
async def test_disallowed_urls_are_not_fetched(url, dns):
    handler = MagicMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(client, url, 1)

    assert result == links.LinkCheckResult(
        url=url, valid=False, error="Not checked: host not allowed"
    )
    handler.assert_not_called()
    dns.assert_not_awaited()


@pytest.mark.parametrize(
    "host", sorted(links.ALLOWED_HOSTS) + ["nested.docs.langchain.com", "github.com"]
)
@pytest.mark.asyncio
async def test_allowed_citations_keep_host_and_sni(host, dns):
    url = (
        f"https://{host}/langchain-ai/langchain"
        if host == "github.com"
        else f"https://{host}/guide"
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="<title>Documentation</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(client, url, 1000)

    assert result.valid
    assert result.final_url is None
    assert len(requests) == 1
    assert requests[0].url.host == PUBLIC_IP
    assert requests[0].headers["host"] == host
    assert requests[0].extensions["sni_hostname"] == host
    assert requests[0].extensions["timeout"]["connect"] == links.DEFAULT_TIMEOUT
    assert requests[0].method == ("GET" if host in links.SOFT_404_DOMAINS else "HEAD")
    dns.assert_awaited_once()


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["169.254.169.254"],
        ["10.0.0.1"],
        ["172.16.0.1"],
        ["192.168.1.1"],
        ["224.0.0.1"],
        ["240.0.0.1"],
        ["0.0.0.0"],
        ["100.64.0.1"],
        ["::1"],
        ["fe80::1"],
        ["fc00::1"],
        ["ff02::1"],
        ["::"],
        ["::ffff:127.0.0.1"],
        [PUBLIC_IP, "127.0.0.1"],
        [],
    ],
)
@pytest.mark.asyncio
async def test_unsafe_or_empty_dns_fails_closed(addresses, dns):
    dns.return_value = _records(*addresses)
    handler = MagicMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/guide", 1
        )

    assert not result.valid
    assert result.final_url is None
    assert "unsafe or unresolved host" in result.error
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_dns_failure_and_timeout_make_no_request(dns):
    handler = MagicMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        for failure in (socket.gaierror(), TimeoutError()):
            links._cache.clear()
            dns.side_effect = failure
            result = await links._check_single_url(
                client, "https://docs.langchain.com/guide", 1
            )
            assert not result.valid
    handler.assert_not_called()


@pytest.mark.parametrize(
    "destination",
    ["https://example.com/", "http://169.254.169.254/", "http://127.0.0.1/"],
)
@pytest.mark.asyncio
async def test_disallowed_redirect_is_not_fetched_or_exposed(destination, dns):
    handler = MagicMock(
        return_value=httpx.Response(302, headers={"Location": destination})
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/guide", 1
        )

    assert not result.valid
    assert result.error == "Not checked: host not allowed"
    assert result.final_url is None
    assert handler.call_count == 1
    dns.assert_awaited_once()


@pytest.mark.asyncio
async def test_redirect_and_head_fallback_revalidate_dns(dns):
    dns.side_effect = [_records(PUBLIC_IP), _records("127.0.0.1")]
    for response in (
        httpx.Response(302, headers={"Location": "/next"}),
        httpx.Response(405),
    ):
        dns.reset_mock(side_effect=True)
        dns.side_effect = [_records(PUBLIC_IP), _records("127.0.0.1")]
        links._cache.clear()
        handler = MagicMock(return_value=response)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await links._check_single_url(
                client, "https://www.langchain.com/guide", 1
            )
        assert not result.valid
        assert result.final_url is None
        assert handler.call_count == 1
        assert dns.await_count == 2


@pytest.mark.asyncio
async def test_relative_redirect_and_head_fallback(dns):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": "/next"})
        return httpx.Response(405 if request.method == "HEAD" else 200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(
            client, "https://www.langchain.com/guide", 1
        )

    assert result.valid
    assert result.final_url == "https://www.langchain.com/next"
    assert [request.method for request in requests] == ["HEAD", "HEAD", "GET"]
    assert dns.await_count == 3


@pytest.mark.asyncio
async def test_redirect_limit(dns):
    handler = MagicMock(return_value=httpx.Response(302, headers={"Location": "/next"}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/guide", 1
        )
    assert result.error == "Too many redirects"
    assert result.final_url is None
    assert handler.call_count == links.MAX_REDIRECTS + 1


@pytest.mark.parametrize("host", sorted(links.SOFT_404_DOMAINS))
@pytest.mark.parametrize(
    "content", ["<title>Page Not Found</title>", "<div>Article Not Found</div>"]
)
@pytest.mark.asyncio
async def test_soft_404_preserved(host, content, dns):
    handler = MagicMock(return_value=httpx.Response(200, text=content))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(client, f"https://{host}/guide", 1)
    assert not result.valid
    assert result.status_code == 200
    assert result.error == "Soft 404: Page shows 'not found' content"


@pytest.mark.asyncio
async def test_redirect_into_docs_uses_soft_404_detection(dns):
    responses = [
        httpx.Response(302, headers={"Location": "https://docs.langchain.com/missing"}),
        httpx.Response(200, text="<title>404</title>"),
    ]
    handler = MagicMock(side_effect=responses)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(
            client, "https://www.langchain.com/guide", 1
        )
    assert not result.valid
    assert result.final_url == "https://docs.langchain.com/missing"
    assert handler.call_args_list[1].args[0].method == "GET"


@pytest.mark.asyncio
async def test_tool_caps_urls_deduplicates_and_clamps_timeout():
    urls = [f"https://docs.langchain.com/{number}" for number in range(20)]
    with patch.object(
        links, "_check_urls_async", new=AsyncMock(return_value=[])
    ) as check:
        await links.check_links.ainvoke({"urls": [urls[0], *urls], "timeout": 1000})
    check.assert_awaited_once_with(urls[:10], links.DEFAULT_TIMEOUT)


@pytest.mark.asyncio
async def test_real_transport_pins_ip_and_preserves_tls_verification(dns, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    dns.side_effect = [_records(PUBLIC_IP), _records("127.0.0.1")]
    stream = MagicMock()
    stream.start_tls = AsyncMock(return_value=stream)
    stream.write = AsyncMock()
    stream.read = AsyncMock(
        return_value=b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
    )
    stream.aclose = AsyncMock()
    stream.get_extra_info.return_value = None
    with patch(
        "httpcore._backends.auto.AutoBackend.connect_tcp",
        new=AsyncMock(return_value=stream),
    ) as connect:
        results = await links._check_urls_async(["https://docs.langchain.com/guide"], 1)

    assert results[0].valid
    dns.assert_awaited_once()
    assert connect.await_args.kwargs["host"] == PUBLIC_IP
    assert connect.await_args.kwargs["port"] == 443
    tls = stream.start_tls.await_args.kwargs
    assert tls["server_hostname"] == "docs.langchain.com"
    assert tls["ssl_context"].check_hostname
    assert tls["ssl_context"].verify_mode == ssl.CERT_REQUIRED
    assert b"Host: docs.langchain.com\r\n" in stream.write.await_args_list[0].args[0]
    stream.aclose.assert_awaited()


@pytest.mark.asyncio
async def test_connection_failure_tries_only_other_validated_addresses(dns):
    dns.return_value = _records(PUBLIC_IP, "1.1.1.1")
    handler = MagicMock(
        side_effect=[httpx.ConnectError("Unavailable"), httpx.Response(200)]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(
            client, "https://www.langchain.com/guide", 1
        )
    assert result.valid
    assert [call.args[0].url.host for call in handler.call_args_list] == [
        PUBLIC_IP,
        "1.1.1.1",
    ]
    dns.assert_awaited_once()


@pytest.mark.asyncio
async def test_public_ipv6_address_is_usable(dns):
    address = "2606:4700:4700::1111"
    dns.return_value = _records(address)
    handler = MagicMock(return_value=httpx.Response(200))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/guide", 1
        )
    assert result.valid
    assert handler.call_args.args[0].url.host == address
    assert handler.call_args.args[0].headers["host"] == "docs.langchain.com"


@pytest.mark.asyncio
async def test_dns_resolution_is_bounded_by_timeout(dns):
    async def slow_lookup(*args, **kwargs):
        await asyncio.sleep(1)
        return _records(PUBLIC_IP)

    dns.side_effect = slow_lookup
    handler = MagicMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/guide", 0.1
        )
    assert result.error == "Request timed out"
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_batch_caps_actual_requests(dns):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    with patch.object(links.httpx, "AsyncClient", return_value=client) as factory:
        results = await links._check_urls_async(
            [f"https://docs.langchain.com/{number}" for number in range(20)], 1
        )
    assert len(results) == 10
    assert dns.await_count == 10
    assert not factory.call_args.kwargs["trust_env"]
    assert not factory.call_args.kwargs["follow_redirects"]
    assert factory.call_args.kwargs["limits"].max_keepalive_connections == 0
