"""Tests for link-check destination restrictions and safe connections."""

import asyncio
import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools import link_check_tools as links


@pytest.fixture(autouse=True)
def clear_link_cache():
    links._cache.clear()
    yield
    links._cache.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/",
        "http://localhost/",
        "http://10.0.0.1/",
        "http://metadata.google.internal/",
        "https://callback.oast.site/",
        "https://webhook.site/anything",
        "http://[::1]/",
        "https://docs.langchain.com.evil.example/",
        "https://docs.langchain.com@webhook.site/",
        "https://user:password@docs.langchain.com/",
        "https://docs.langchain.com:8080/",
        "https://github.com/other-org/repo",
        "https://github.com/langchain-ai/../other-org",
        "https://github.com/langchain-ai/%2e%2e/other-org",
        "https://docs.langchain.com\\@webhook.site/",
        "ftp://docs.langchain.com/",
    ],
)
async def test_disallowed_urls_make_no_http_or_dns_request(url):
    with (
        patch.object(
            httpx.AsyncHTTPTransport, "handle_async_request", new_callable=AsyncMock
        ) as send,
        patch.object(
            asyncio.get_running_loop(), "getaddrinfo", new_callable=AsyncMock
        ) as resolve,
    ):
        result = (await links._check_urls_async([url], timeout=1.0))[0]

    assert not result.valid
    assert result.error == links.DOMAIN_NOT_ALLOWED
    send.assert_not_awaited()
    resolve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", ["http://127.0.0.1/", "http://169.254.169.254/", "https://webhook.site/"]
)
async def test_redirect_to_disallowed_destination_is_not_requested(target):
    requested = []

    def respond(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": target})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/start", 1.0
        )

    assert requested == ["https://docs.langchain.com/start"]
    assert result.error == links.DOMAIN_NOT_ALLOWED
    assert not result.valid


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [f"https://{host}/" for host in links.ALLOWED_DOMAINS if host != "github.com"]
    + ["https://github.com/langchain-ai/langchain"],
)
async def test_allowed_urls_are_checked_normally(url):
    requested = []

    def respond(request):
        requested.append(request.method)
        return httpx.Response(200, text="<title>Documentation</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await links._check_single_url(client, url, 1.0)

    assert result.valid
    assert requested == ["GET" if links._needs_soft_404_check(url) else "HEAD"]


@pytest.mark.asyncio
async def test_relative_redirect_and_head_fallback_are_checked():
    requested = []

    def respond(request):
        requested.append((request.method, request.url.path))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/end"})
        if request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await links._check_single_url(
            client, "https://smith.langchain.com/start", 1.0
        )

    assert result.valid
    assert result.final_url == "https://smith.langchain.com/end"
    assert requested == [("HEAD", "/start"), ("HEAD", "/end"), ("GET", "/end")]


@pytest.mark.asyncio
async def test_redirect_limit_is_enforced():
    send = []

    def respond(request):
        send.append(request.url)
        return httpx.Response(302, headers={"location": "/loop"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/loop", 1.0
        )

    assert len(send) == links.MAX_REDIRECTS + 1
    assert result.error == links.UNREACHABLE


def dns_addresses(*addresses):
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "100.64.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
    ],
)
async def test_any_non_public_dns_answer_blocks_connection(address):
    with (
        patch.object(
            asyncio.get_running_loop(),
            "getaddrinfo",
            new_callable=AsyncMock,
            return_value=dns_addresses("93.184.216.34", address),
        ),
        patch.object(
            httpx.AsyncHTTPTransport, "handle_async_request", new_callable=AsyncMock
        ) as send,
    ):
        result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[
            0
        ]

    assert result.error == links.UNREACHABLE
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_address_is_pinned_with_original_host_and_tls_name():
    with (
        patch.object(
            asyncio.get_running_loop(),
            "getaddrinfo",
            new_callable=AsyncMock,
            return_value=dns_addresses("93.184.216.34"),
        ) as resolve,
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, text="Documentation"),
        ) as send,
    ):
        result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[
            0
        ]

    assert result.valid
    resolve.assert_awaited_once()
    request = send.call_args.args[0]
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "docs.langchain.com"
    assert request.extensions["sni_hostname"] == "docs.langchain.com"


@pytest.mark.asyncio
async def test_allowed_redirect_host_with_private_dns_is_blocked():
    with (
        patch.object(
            asyncio.get_running_loop(),
            "getaddrinfo",
            new_callable=AsyncMock,
            side_effect=[dns_addresses("93.184.216.34"), dns_addresses("127.0.0.1")],
        ) as resolve,
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            new_callable=AsyncMock,
            return_value=httpx.Response(
                302, headers={"location": "https://support.langchain.com/"}
            ),
        ) as send,
    ):
        result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[
            0
        ]

    assert not result.valid
    assert result.error == links.UNREACHABLE
    assert resolve.await_count == 2
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_failures_hide_exception_details():
    def respond(request):
        raise httpx.ConnectError("private server connection details", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await links._check_single_url(
            client, "https://docs.langchain.com/", 1.0
        )

    assert result.error == links.UNREACHABLE
    assert "private server" not in links._format_results([result])


def test_formatter_never_exposes_raw_errors_or_status_codes():
    result = links.LinkCheckResult(
        "https://docs.langchain.com/",
        False,
        status_code=404,
        error="HTTP 404: private details",
    )
    output = links._format_results([result])

    assert links.UNREACHABLE in output
    assert "404" not in output
    assert "private details" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [[], socket.gaierror("private resolver details")])
async def test_dns_failures_make_no_connection(answer):
    resolve = AsyncMock()
    if isinstance(answer, Exception):
        resolve.side_effect = answer
    else:
        resolve.return_value = answer
    with (
        patch.object(asyncio.get_running_loop(), "getaddrinfo", new=resolve),
        patch.object(
            httpx.AsyncHTTPTransport, "handle_async_request", new_callable=AsyncMock
        ) as send,
    ):
        result = (await links._check_urls_async(["https://docs.langchain.com/"], 1.0))[
            0
        ]

    assert result.error == links.UNREACHABLE
    send.assert_not_awaited()
