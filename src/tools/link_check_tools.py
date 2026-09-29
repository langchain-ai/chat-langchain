"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import os
import re
import socket
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection
CACHE_MAX_SIZE = 256
ALLOWED_HOSTS = {
    host.strip().lower()
    for host in os.getenv(
        "ALLOWED_HOSTS",
        "docs.langchain.com,python.langchain.com,js.langchain.com,support.langchain.com,"
        "blog.langchain.com,langchain.com,github.com",
    ).split(",")
    if host.strip()
}

SOFT_404_DOMAINS = {
    "docs.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "support.langchain.com",
}

_cache: OrderedDict[str, "LinkCheckResult"] = OrderedDict()
_safe_url_addresses: OrderedDict[str, str] = OrderedDict()


@dataclass
class LinkCheckResult:
    """Result of checking a single URL."""

    url: str
    valid: bool
    status_code: int | None = None
    error: str | None = None
    final_url: str | None = None


def _is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def _resolve_safe_url(url: str) -> str | None:
    """Resolve a URL to an allowed, public IP address."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        hostname = parsed.hostname.lower() if parsed.hostname else None
        if not hostname or not any(
            hostname == allowed or hostname.endswith(f".{allowed}")
            for allowed in ALLOWED_HOSTS
        ):
            return None
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            return None

        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
        for address in addresses:
            parsed_address = ipaddress.ip_address(address)
            if any(
                (
                    parsed_address.is_private,
                    parsed_address.is_loopback,
                    parsed_address.is_link_local,
                    parsed_address.is_reserved,
                    parsed_address.is_multicast,
                    parsed_address.is_unspecified,
                )
            ):
                return None
        return next(iter(addresses), None)
    except (OSError, ValueError):
        return None
    return None


def _is_safe_url(url: str) -> bool:
    """Check whether a URL targets an allowed public host."""
    address = _resolve_safe_url(url)
    if address is None:
        return False
    _safe_url_addresses[url] = address
    _safe_url_addresses.move_to_end(url)
    while len(_safe_url_addresses) > CACHE_MAX_SIZE:
        _safe_url_addresses.popitem(last=False)
    return True


def _cache_result(url: str, result: "LinkCheckResult") -> None:
    """Store a result while keeping the cache bounded."""
    _cache[url] = result
    _cache.move_to_end(url)
    while len(_cache) > CACHE_MAX_SIZE:
        _cache.popitem(last=False)


class _PinnedAsyncHTTPTransport(httpx.AsyncBaseTransport):
    """Send requests to the address checked before the request."""

    def __init__(self) -> None:
        self._transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        address = request.extensions.pop("resolved_address")
        parsed = urlparse(str(request.url))
        pinned_url = request.url.copy_with(host=address)
        headers = dict(request.headers)
        headers["host"] = parsed.netloc
        extensions = dict(request.extensions)
        extensions["sni_hostname"] = parsed.hostname
        pinned_request = httpx.Request(
            request.method,
            pinned_url,
            headers=headers,
            content=request.stream,
            extensions=extensions,
        )
        return await self._transport.handle_async_request(pinned_request)

    async def aclose(self) -> None:
        await self._transport.aclose()


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = urlparse(url).hostname.lower()
        return domain in SOFT_404_DOMAINS
    except (AttributeError, ValueError):
        return False


def _is_soft_404(content: str) -> bool:
    """Detect soft 404 pages that return HTTP 200 but show 'not found' content."""
    if "Article Not Found" in content:
        return True

    title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if any(phrase in title for phrase in ["not found", "404", "page not found"]):
            return True
    return False


async def _check_single_url(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
) -> LinkCheckResult:
    """Check a single URL for validity."""
    if url in _cache:
        _cache.move_to_end(url)
        return _cache[url]

    if not _is_valid_url(url):
        result = LinkCheckResult(url=url, valid=False, error="Invalid URL format")
        _cache_result(url, result)
        return result

    if not _is_safe_url(url):
        result = LinkCheckResult(url=url, valid=False, error="URL not permitted")
        _cache_result(url, result)
        return result
    resolved_address = _safe_url_addresses.pop(url, None)
    if resolved_address is None:
        result = LinkCheckResult(url=url, valid=False, error="URL not permitted")
        _cache_result(url, result)
        return result

    try:
        current_url = url
        final_url = None
        for redirect_count in range(MAX_REDIRECTS + 1):
            if current_url != url:
                if not _is_safe_url(current_url):
                    result = LinkCheckResult(
                        url=url, valid=False, error="URL not permitted"
                    )
                    _cache_result(url, result)
                    return result
                resolved_address = _safe_url_addresses.pop(current_url, None)
            if resolved_address is None:
                result = LinkCheckResult(
                    url=url, valid=False, error="URL not permitted"
                )
                _cache_result(url, result)
                return result

            method = "GET" if _needs_soft_404_check(current_url) else "HEAD"
            request = client.build_request(method, current_url, timeout=timeout)
            request.extensions["resolved_address"] = resolved_address
            response = await client.send(request, stream=True)
            try:
                if response.status_code == 405 and method == "HEAD":
                    await response.aclose()
                    request = client.build_request("GET", current_url, timeout=timeout)
                    request.extensions["resolved_address"] = resolved_address
                    response = await client.send(request, stream=True)

                if 300 <= response.status_code < 400:
                    location = response.headers.get("location")
                    await response.aclose()
                    if not location:
                        break
                    if redirect_count >= MAX_REDIRECTS:
                        result = LinkCheckResult(
                            url=url, valid=False, error="Too many redirects"
                        )
                        _cache_result(url, result)
                        return result
                    current_url = urljoin(current_url, location)
                    if not _is_valid_url(current_url) or not _is_safe_url(current_url):
                        result = LinkCheckResult(
                            url=url, valid=False, error="URL not permitted"
                        )
                        _cache_result(url, result)
                        return result
                    final_url = current_url
                    continue

                is_valid = 200 <= response.status_code < 400
                if (
                    is_valid
                    and response.status_code == 200
                    and _needs_soft_404_check(current_url)
                ):
                    content = ""
                    async for chunk in response.aiter_text():
                        content += chunk
                        if len(content) >= CONTENT_CHECK_BYTES:
                            break

                    if _is_soft_404(content):
                        result = LinkCheckResult(
                            url=url,
                            valid=False,
                            status_code=200,
                            final_url=final_url,
                            error="Soft 404: Page shows 'not found' content",
                        )
                        _cache_result(url, result)
                        return result

                result = LinkCheckResult(
                    url=url,
                    valid=is_valid,
                    status_code=response.status_code,
                    final_url=final_url,
                    error=None if is_valid else f"HTTP {response.status_code}",
                )
                _cache_result(url, result)
                return result
            finally:
                await response.aclose()

        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")

    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError as e:
        result = LinkCheckResult(
            url=url, valid=False, error=f"Connection failed: {str(e)[:50]}"
        )
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        result = LinkCheckResult(url=url, valid=False, error=f"Error: {str(e)[:50]}")

    _cache_result(url, result)
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        transport=_PinnedAsyncHTTPTransport(),
    ) as client:
        tasks = [_check_single_url(client, url, timeout) for url in urls]
        return list(await asyncio.gather(*tasks))


def _format_results(results: list[LinkCheckResult]) -> str:
    """Format check results into readable output."""
    if not results:
        return "No URLs to check."

    valid = [r for r in results if r.valid]
    invalid = [r for r in results if not r.valid]

    lines = [f"Link Check Results: {len(valid)}/{len(results)} valid\n"]

    if invalid:
        lines.append("Invalid links:")
        lines.extend(f"  - {r.url}: {r.error}" for r in invalid)
        lines.append("")

    if valid:
        lines.append("Valid links:")
        for r in valid:
            suffix = f" (→ {r.final_url})" if r.final_url else ""
            lines.append(f"  - {r.url}{suffix}")

    return "\n".join(lines)


@tool
async def check_links(urls: list[str], timeout: float = DEFAULT_TIMEOUT) -> str:
    """Check if URLs are valid and accessible before including them in a response.

    Args:
        urls: List of URLs to validate.
        timeout: Timeout per request in seconds (default: 10).

    Returns:
        Formatted results showing which URLs are valid/invalid with details.
    """
    if not urls:
        return "No URLs provided to check."

    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
