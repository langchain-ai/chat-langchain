"""Tests for link-check destination enforcement without real network requests."""

import socket
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.tools import link_check_tools as links


def _addresses(*addresses):
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


@pytest.fixture
def network(monkeypatch):
    links._cache.clear()
    resolver = Mock(return_value=_addresses("8.8.8.8"))
    sender = AsyncMock(
        side_effect=lambda request: httpx.Response(
            200, text="<title>Documentation</title>"
        )
    )
    monkeypatch.setattr(links.socket, "getaddrinfo", resolver)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", sender)
    yield resolver, sender
    links._cache.clear()


def _assert_blocked(result, error="Skipped host"):
    assert not result.valid
    assert error in result.error
    assert result.status_code is None
    assert result.final_url is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://example.com/",
        "https://docs.langchain.com.attacker.com/",
        "https://notlangchain.com/",
        "https://github.com.attacker.com/",
        "https://attacker.github.com/",
    ],
)
async def test_disallowed_hosts_are_rejected_before_cache_or_requests(network, url):
    resolver, sender = network
    links._cache[url] = links.LinkCheckResult(url=url, valid=True, status_code=200)

    result = (await links._check_urls_async([url], timeout=1.0))[0]

    _assert_blocked(result)
    resolver.assert_not_called()
    sender.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "ftp://docs.langchain.com/",
        "https://docs.langchain.com:invalid/",
        "https://docs.langchain.com:65536/",
        "https://docs.langchain.com:0/",
        "https://user:password@docs.langchain.com/",
        "https://@docs.langchain.com/",
        "https://docs.langchain.com\n.attacker.com/",
    ],
)
async def test_malformed_urls_are_rejected_without_requests(network, url):
    resolver, sender = network

    result = (await links._check_urls_async([url], timeout=1.0))[0]

    _assert_blocked(result, "Invalid URL format")
    resolver.assert_not_called()
    sender.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "240.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "100.64.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::",
        "2001:db8::1",
        "::ffff:127.0.0.1",
    ],
)
async def test_allowlisted_hosts_with_prohibited_dns_addresses_are_blocked(
    network, address
):
    resolver, sender = network
    resolver.return_value = _addresses(address)

    result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[0]

    _assert_blocked(result)
    assert address not in result.error
    sender.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("addresses", [[], _addresses("8.8.8.8", "10.0.0.1")])
async def test_empty_or_mixed_dns_answers_fail_closed(network, addresses):
    resolver, sender = network
    resolver.return_value = addresses

    result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[0]

    _assert_blocked(result)
    sender.assert_not_awaited()


@pytest.mark.asyncio
async def test_dns_resolution_failure_does_not_expose_raw_errors(network):
    resolver, sender = network
    resolver.side_effect = socket.gaierror("internal DNS details")

    result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[0]

    _assert_blocked(result)
    assert "internal DNS details" not in result.error
    sender.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hostname",
    [
        "docs.langchain.com",
        "support.langchain.com",
        "reference.langchain.com",
        "python.langchain.com",
        "nested.docs.langchain.com",
        "langchain.com",
        "langchain-ai.github.io",
        "github.com",
    ],
)
async def test_allowed_hosts_connect_to_validated_ip_with_original_host_and_sni(
    network, monkeypatch, hostname
):
    resolver, sender = network
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")

    result = (await links._check_urls_async([f"https://{hostname}/"], 1.0))[0]

    assert result.valid
    assert result.status_code == 200
    resolver.assert_called_once_with(hostname, 443, type=socket.SOCK_STREAM)
    request = sender.call_args.args[0]
    assert request.url.host == "8.8.8.8"
    assert request.headers["Host"] == hostname
    assert request.extensions["sni_hostname"] == hostname


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    ["http://10.0.0.1/private", "http://169.254.169.254/", "https://attacker.com/"],
)
async def test_redirect_to_disallowed_host_is_blocked_before_redirected_request(
    network, location
):
    resolver, sender = network
    sender.side_effect = [httpx.Response(302, headers={"Location": location})]

    result = (await links._check_urls_async(["https://docs.langchain.com/start"], 1.0))[
        0
    ]

    _assert_blocked(result)
    assert resolver.call_count == 1
    assert sender.await_count == 1
    assert location not in links._format_results([result])


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_dns_is_revalidated_for_redirects_and_head_to_get_fallback(
    network, fallback
):
    resolver, sender = network
    resolver.side_effect = [_addresses("8.8.8.8"), _addresses("127.0.0.1")]
    sender.side_effect = [
        httpx.Response(405)
        if fallback
        else httpx.Response(302, headers={"Location": "/next"})
    ]

    result = (await links._check_urls_async(["https://github.com/start"], 1.0))[0]

    _assert_blocked(result)
    assert resolver.call_count == 2
    assert sender.await_count == 1


@pytest.mark.asyncio
async def test_allowed_relative_redirect_uses_new_validated_address(network):
    resolver, sender = network
    resolver.side_effect = [_addresses("8.8.8.8"), _addresses("1.1.1.1")]
    sender.side_effect = [
        httpx.Response(302, headers={"Location": "/next"}),
        httpx.Response(200, text="<title>Documentation</title>"),
    ]

    result = (await links._check_urls_async(["https://docs.langchain.com/start"], 1.0))[
        0
    ]

    assert result.valid
    assert result.final_url == "https://docs.langchain.com/next"
    assert [call.args[0].url.host for call in sender.call_args_list] == [
        "8.8.8.8",
        "1.1.1.1",
    ]
    assert all(
        call.args[0].headers["Host"] == "docs.langchain.com"
        for call in sender.call_args_list
    )


@pytest.mark.asyncio
async def test_public_ipv6_is_pinned_without_changing_host_or_sni(network):
    resolver, sender = network
    resolver.return_value = _addresses("2606:4700:4700::1111")

    result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[0]

    assert result.valid
    request = sender.call_args.args[0]
    assert request.url.host == "2606:4700:4700::1111"
    assert request.headers["Host"] == "docs.langchain.com"
    assert request.extensions["sni_hostname"] == "docs.langchain.com"


@pytest.mark.asyncio
async def test_head_to_get_fallback_remains_supported(network):
    resolver, sender = network
    sender.side_effect = [httpx.Response(405), httpx.Response(200)]

    result = (await links._check_urls_async(["https://github.com/langchain-ai"], 1.0))[
        0
    ]

    assert result.valid
    assert resolver.call_count == 2
    assert [call.args[0].method for call in sender.call_args_list] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_redirect_after_head_to_get_fallback_is_revalidated(network):
    resolver, sender = network
    sender.side_effect = [
        httpx.Response(405),
        httpx.Response(302, headers={"Location": "http://127.0.0.1/private"}),
    ]

    result = (await links._check_urls_async(["https://github.com/langchain-ai"], 1.0))[
        0
    ]

    _assert_blocked(result)
    assert resolver.call_count == 2
    assert sender.await_count == 2


@pytest.mark.asyncio
async def test_redirect_from_head_to_docs_still_detects_soft_404(network):
    _, sender = network
    sender.side_effect = [
        httpx.Response(302, headers={"Location": "https://docs.langchain.com/missing"}),
        httpx.Response(200, text="<title>404 Not Found</title>"),
    ]

    result = (await links._check_urls_async(["https://github.com/langchain-ai"], 1.0))[
        0
    ]

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert [call.args[0].method for call in sender.call_args_list] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_redirect_limit_is_enforced(network):
    _, sender = network
    sender.side_effect = lambda request: httpx.Response(
        302, headers={"Location": "/loop"}
    )

    result = (await links._check_urls_async(["https://docs.langchain.com/start"], 1.0))[
        0
    ]

    _assert_blocked(result, "Too many redirects")
    assert sender.await_count == links.MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_connection_errors_do_not_expose_destination_details(network):
    _, sender = network
    sender.side_effect = httpx.ConnectError("raw connection details")

    result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[0]

    _assert_blocked(result, "Connection failed")
    assert "raw connection details" not in links._format_results([result])
