"""Test link checking host restrictions without real DNS or HTTP requests."""

import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools.link_check_tools import (
    ALLOWED_HOSTS,
    DEFAULT_TIMEOUT,
    HOST_NOT_ALLOWED,
    MAX_REDIRECTS,
    LinkCheckResult,
    _cache,
    _check_urls_async,
    _format_results,
    _is_valid_url,
    check_links,
)


@pytest.fixture(autouse=True)
def mock_dns():
    _cache.clear()
    with patch(
        "asyncio.BaseEventLoop.getaddrinfo",
        new=AsyncMock(
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
        ),
    ) as resolver:
        yield resolver
    _cache.clear()


@pytest.fixture
def mock_http():
    requests = []
    responses = []
    client_type = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        return (
            responses.pop(0) if responses else httpx.Response(200, text="Useful docs")
        )

    with patch(
        "src.tools.link_check_tools.httpx.AsyncClient",
        side_effect=lambda **kwargs: client_type(
            transport=httpx.MockTransport(respond), **kwargs
        ),
    ) as factory:
        yield requests, responses, factory


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/",
        "http://metadata.google.internal/",
        "https://abc.oast.site/",
        "http://localhost/",
        "http://[::1]/",
        "https://docs.langchain.com.abc.oast.site/",
        "https://docs.langchain.com@abc.oast.site/",
        "https://abc.oast.site:443/",
        "https://docs.langchain.com:invalid/",
        "ftp://docs.langchain.com/",
        "not-a-url",
    ],
)
@pytest.mark.asyncio
async def test_disallowed_urls_are_rejected_without_requests(url, mock_dns, mock_http):
    requests, _, _ = mock_http
    _cache[url] = LinkCheckResult(url=url, valid=True, status_code=200)

    result = (await _check_urls_async([url], DEFAULT_TIMEOUT))[0]

    assert not result.valid
    assert result.error == HOST_NOT_ALLOWED
    assert result.status_code is None
    assert result.final_url is None
    assert not requests
    mock_dns.assert_not_awaited()
    assert "Rejected links (not checked):" in _format_results([result])
    assert "Invalid links:" not in _format_results([result])
    assert "HTTP" not in _format_results([result])


@pytest.mark.parametrize("host", sorted(ALLOWED_HOSTS))
@pytest.mark.asyncio
async def test_allowed_hosts_pass(host, mock_http):
    requests, _, factory = mock_http
    url = f"https://{host.upper()}:443/docs"

    result = (await _check_urls_async([url], DEFAULT_TIMEOUT))[0]

    assert _is_valid_url(url)
    assert result.valid
    assert result.status_code == 200
    assert len(requests) == 1
    assert factory.call_args.kwargs["follow_redirects"] is False


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "240.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "100.64.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::",
        "::ffff:127.0.0.1",
    ],
)
@pytest.mark.asyncio
async def test_any_non_public_dns_answer_is_rejected(address, mock_dns, mock_http):
    requests, _, _ = mock_http
    mock_dns.return_value.append(
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 443),
        )
    )
    url = "https://docs.langchain.com/"

    result = (await _check_urls_async([url], DEFAULT_TIMEOUT))[0]

    assert not result.valid
    assert result.error == HOST_NOT_ALLOWED
    assert result.status_code is None
    assert not requests
    assert url not in _cache


@pytest.mark.parametrize("answers", [[], OSError("DNS failed"), TimeoutError()])
@pytest.mark.asyncio
async def test_dns_failures_are_rejected(answers, mock_dns, mock_http):
    requests, _, _ = mock_http
    if isinstance(answers, Exception):
        mock_dns.side_effect = answers
    else:
        mock_dns.return_value = answers

    result = (
        await _check_urls_async(["https://docs.langchain.com/"], DEFAULT_TIMEOUT)
    )[0]

    assert result.error == HOST_NOT_ALLOWED
    assert not requests
    assert not _cache


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "https://abc.oast.site/",
        "//metadata.google.internal/",
        "http://[invalid/",
    ],
)
@pytest.mark.asyncio
async def test_redirect_to_disallowed_host_is_not_followed(target, mock_dns, mock_http):
    requests, responses, _ = mock_http
    responses.append(httpx.Response(302, headers={"location": target}))
    url = "https://docs.langchain.com/"

    result = (await _check_urls_async([url], DEFAULT_TIMEOUT))[0]

    assert result.error == HOST_NOT_ALLOWED
    assert result.status_code is None
    assert result.final_url is None
    assert [str(request.url) for request in requests] == [url]
    assert mock_dns.await_count == 1
    assert url not in _cache


@pytest.mark.asyncio
async def test_redirect_dns_is_checked_again(mock_dns, mock_http):
    requests, responses, _ = mock_http
    responses.append(httpx.Response(302, headers={"location": "/internal"}))
    mock_dns.side_effect = [
        mock_dns.return_value,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]

    result = (
        await _check_urls_async(["https://docs.langchain.com/"], DEFAULT_TIMEOUT)
    )[0]

    assert result.error == HOST_NOT_ALLOWED
    assert len(requests) == 1
    assert mock_dns.await_count == 2
    assert not _cache


@pytest.mark.asyncio
async def test_allowed_redirects_and_soft_404_checks(mock_dns, mock_http):
    requests, responses, _ = mock_http
    responses.extend(
        [
            httpx.Response(
                301, headers={"location": "https://docs.langchain.com/start"}
            ),
            httpx.Response(302, headers={"location": "/missing"}),
            httpx.Response(200, text="<title>Page not found</title>"),
        ]
    )

    result = (await _check_urls_async(["https://github.com/"], DEFAULT_TIMEOUT))[0]

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert result.final_url == "https://docs.langchain.com/missing"
    assert [request.method for request in requests] == ["HEAD", "GET", "GET"]
    assert mock_dns.await_count == 3


@pytest.mark.asyncio
async def test_redirect_limit(mock_http):
    requests, responses, _ = mock_http
    responses.extend(
        httpx.Response(302, headers={"location": f"/{index}"})
        for index in range(MAX_REDIRECTS + 1)
    )

    result = (
        await _check_urls_async(["https://docs.langchain.com/"], DEFAULT_TIMEOUT)
    )[0]

    assert result.error == "Too many redirects"
    assert len(requests) == MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_head_fallback_and_timeout_cap(mock_dns, mock_http):
    requests, responses, _ = mock_http
    responses.append(httpx.Response(405))

    output = await check_links.ainvoke(
        {"urls": ["https://github.com/"], "timeout": 1000}
    )

    assert "1/1 valid" in output
    assert [request.method for request in requests] == ["HEAD", "GET"]
    assert mock_dns.await_count == 2
    assert all(
        timeout == DEFAULT_TIMEOUT
        for request in requests
        for timeout in request.extensions["timeout"].values()
    )
