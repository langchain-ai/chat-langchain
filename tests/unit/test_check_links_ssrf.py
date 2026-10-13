"""Test citation URL restrictions without making network requests."""

import socket
import threading
from unittest.mock import Mock

import httpx
import pytest

from src.tools import link_check_tools


def _addresses(*addresses):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))
        for address in addresses
    ]


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    link_check_tools._cache.clear()
    resolver = Mock(return_value=_addresses("93.184.216.34"))
    monkeypatch.setattr(link_check_tools.socket, "getaddrinfo", resolver)
    yield resolver
    link_check_tools._cache.clear()


async def _check(url, handler):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        return await link_check_tools._check_single_url(client, url, timeout=1.0)


@pytest.mark.asyncio
async def test_allowed_docs_url_passes(public_dns):
    handler = Mock(return_value=httpx.Response(200, text="Documentation"))
    result = await _check("https://docs.langchain.com/overview", handler)

    assert result.valid
    assert result.status_code == 200
    assert handler.call_count == 1
    assert handler.call_args.args[0].method == "GET"
    public_dns.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "https://docs.langchain.com.example.com/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://metadata.google.internal/",
        "https://user:password@docs.langchain.com/",
        "https://docs.langchain.com:8080/",
        "https://github.com/other-org/repo",
        "https://github.com/langchain-ai/../other-org/repo",
        "https://github.com/langchain-ai/%2e%2e/other-org/repo",
    ],
)
async def test_disallowed_url_is_cached_without_network_call(url, public_dns):
    handler = Mock(side_effect=AssertionError("Unexpected HTTP request"))
    result = await _check(url, handler)

    assert not result.valid
    assert result.error == "Host not allowed"
    assert result.final_url is None
    assert link_check_tools._cache[url] is result
    assert await _check(url, handler) is result
    handler.assert_not_called()
    public_dns.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.2",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "100.64.0.1",
        "240.0.0.1",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
async def test_allowed_host_with_non_public_dns_is_rejected(address, public_dns):
    public_dns.return_value = _addresses("93.184.216.34", address)
    handler = Mock(side_effect=AssertionError("Unexpected HTTP request"))
    result = await _check("https://docs.langchain.com/overview", handler)

    assert not result.valid
    assert result.error == "Host not allowed"
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_dns_resolution_runs_off_event_loop(public_dns):
    event_loop_thread = threading.get_ident()

    def resolve(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        return _addresses("93.184.216.34")

    public_dns.side_effect = resolve
    result = await _check(
        "https://reference.langchain.com/", lambda request: httpx.Response(200)
    )

    assert result.valid


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/",
        "https://example.com/callback",
        "//example.com/callback",
        "https://github.com/other-org/repo",
        "ftp://docs.langchain.com/",
    ],
)
@pytest.mark.parametrize("host", ["docs.langchain.com", "reference.langchain.com"])
async def test_redirect_to_disallowed_host_is_not_requested(location, host, public_dns):
    handler = Mock(return_value=httpx.Response(302, headers={"location": location}))
    result = await _check(f"https://{host}/start", handler)

    assert not result.valid
    assert result.error == "Redirect to disallowed host"
    assert result.final_url is None
    assert result.status_code is None
    assert handler.call_count == 1
    public_dns.assert_called_once()


@pytest.mark.asyncio
async def test_redirect_target_dns_is_checked_again(public_dns):
    public_dns.side_effect = [
        _addresses("93.184.216.34"),
        _addresses("169.254.169.254"),
    ]
    handler = Mock(return_value=httpx.Response(302, headers={"location": "/next"}))
    result = await _check("https://docs.langchain.com/start", handler)

    assert not result.valid
    assert result.error == "Redirect to disallowed host"
    assert result.final_url is None
    assert handler.call_count == 1
    assert public_dns.call_count == 2


@pytest.mark.asyncio
async def test_relative_redirect_and_final_destination_soft_404():
    handler = Mock(
        side_effect=[
            httpx.Response(
                302, headers={"location": "https://docs.langchain.com/start"}
            ),
            httpx.Response(301, headers={"location": "/missing"}),
            httpx.Response(200, text="<title>Page not found</title>"),
        ]
    )
    result = await _check("https://reference.langchain.com/start", handler)

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert result.final_url == "https://docs.langchain.com/missing"
    assert [call.args[0].method for call in handler.call_args_list] == [
        "HEAD",
        "GET",
        "GET",
    ]


@pytest.mark.asyncio
async def test_allowed_redirect_returns_final_url():
    handler = Mock(
        side_effect=[
            httpx.Response(302, headers={"location": "/next"}),
            httpx.Response(200, text="Documentation"),
        ]
    )
    result = await _check("https://docs.langchain.com/start", handler)

    assert result.valid
    assert result.final_url == "https://docs.langchain.com/next"


@pytest.mark.asyncio
async def test_redirect_limit_is_enforced():
    handler = Mock(return_value=httpx.Response(302, headers={"location": "/loop"}))
    result = await _check("https://docs.langchain.com/start", handler)

    assert not result.valid
    assert result.error == "Too many redirects"
    assert result.final_url is None
    assert handler.call_count == link_check_tools.MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_github_org_and_head_fallback():
    handler = Mock(side_effect=[httpx.Response(405), httpx.Response(200)])
    result = await _check("https://github.com/langchain-ai/langchain", handler)

    assert result.valid
    assert [call.args[0].method for call in handler.call_args_list] == ["HEAD", "GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (httpx.ConnectError("private resolver details"), "Connection failed"),
        (RuntimeError("private connection details"), "Request error"),
    ],
)
async def test_request_errors_are_generic(exception, error):
    handler = Mock(side_effect=exception)
    result = await _check("https://docs.langchain.com/overview", handler)

    assert not result.valid
    assert result.error == error
    assert "private" not in link_check_tools._format_results([result])


@pytest.mark.asyncio
async def test_dns_errors_are_generic(public_dns):
    public_dns.side_effect = socket.gaierror("private DNS details")
    handler = Mock(side_effect=AssertionError("Unexpected HTTP request"))
    result = await _check("https://docs.langchain.com/overview", handler)

    assert result.error == "Connection failed"
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_batch_client_disables_automatic_redirects(monkeypatch):
    original_client = httpx.AsyncClient
    handler = Mock(
        return_value=httpx.Response(
            302, headers={"location": "http://169.254.169.254/"}
        )
    )

    def client_factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(link_check_tools.httpx, "AsyncClient", client_factory)
    results = await link_check_tools._check_urls_async(
        ["https://docs.langchain.com/start"], timeout=1.0
    )

    assert results[0].error == "Redirect to disallowed host"
    assert handler.call_count == 1
