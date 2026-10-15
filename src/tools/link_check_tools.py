"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection

# Domains known to have soft 404s (return 200 with "not found" content)
SOFT_404_DOMAINS = {
    "docs.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "support.langchain.com",
}

# Simple in-memory cache
_cache: dict[str, "LinkCheckResult"] = {}


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
        result = httpx.URL(url)
        return result.scheme in ("http", "https") and bool(result.host)
    except (httpx.InvalidURL, ValueError):
        return False


def _is_public_ip(value: str) -> bool:
    """Check whether an address is eligible for public HTTP requests."""
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_reserved,
            address.is_multicast,
            address.is_unspecified,
        )
    )


async def _is_allowed_destination(url: str) -> bool:
    """Allow only HTTP(S) destinations resolving entirely to public addresses."""
    if not _is_valid_url(url):
        return False
    parsed = httpx.URL(url)
    hostname = parsed.host.rstrip(".").lower()
    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname == "metadata.google.internal"
        or "%" in hostname
    ):
        return False
    try:
        return _is_public_ip(hostname)
    except ValueError:
        if "." not in hostname:
            return False
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
        return bool(addresses) and all(
            _is_public_ip(str(address[4][0])) for address in addresses
        )
    except (OSError, ValueError):
        return False


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = urlparse(url).netloc.lower()
        return domain in SOFT_404_DOMAINS
    except Exception:
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
    try:
        if not await _is_allowed_destination(url):
            return LinkCheckResult(url=url, valid=False, error="Host not allowed")
        if url in _cache:
            cached = _cache[url]
            if cached.final_url and not await _is_allowed_destination(cached.final_url):
                return LinkCheckResult(url=url, valid=False, error="Host not allowed")
            return cached

        current_url = url
        needs_content_check = _needs_soft_404_check(url)
        method = "GET" if needs_content_check else "HEAD"
        redirects = 0
        while True:
            async with client.stream(
                method, current_url, timeout=timeout, follow_redirects=False
            ) as response:
                if (
                    response.status_code in (301, 302, 303, 307, 308)
                    and "location" in response.headers
                ):
                    destination = urljoin(
                        str(response.url), response.headers["location"]
                    )
                    if not await _is_allowed_destination(destination):
                        return LinkCheckResult(
                            url=url, valid=False, error="Host not allowed"
                        )
                    if redirects >= MAX_REDIRECTS:
                        raise httpx.TooManyRedirects("Too many redirects")
                    redirects += 1
                    current_url = destination
                    needs_content_check = needs_content_check or _needs_soft_404_check(
                        current_url
                    )
                    if needs_content_check:
                        method = "GET"
                    continue

                if method == "HEAD" and response.status_code == 405:
                    if not await _is_allowed_destination(current_url):
                        return LinkCheckResult(
                            url=url, valid=False, error="Host not allowed"
                        )
                    method = "GET"
                    continue

                final_url = str(response.url) if str(response.url) != url else None
                is_valid = 200 <= response.status_code < 400

                if (
                    method == "GET"
                    and needs_content_check
                    and response.status_code == 200
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
                        _cache[url] = result
                        return result

                result = LinkCheckResult(
                    url=url,
                    valid=is_valid,
                    status_code=response.status_code,
                    final_url=final_url,
                    error=None if is_valid else f"HTTP {response.status_code}",
                )
                _cache[url] = result
                return result

    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError:
        result = LinkCheckResult(url=url, valid=False, error="Connection failed")
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        result = LinkCheckResult(url=url, valid=False, error="Request failed")

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        max_redirects=MAX_REDIRECTS,
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
        for result in invalid:
            error = result.error or "Request failed"
            if error not in (
                "Host not allowed",
                "Invalid URL format",
                "Request timed out",
                "Too many redirects",
                "Connection failed",
                "Soft 404: Page shows 'not found' content",
            ) and not re.fullmatch(r"HTTP [1-5][0-9]{2}", error):
                error = "Request failed"
            lines.append(f"  - {result.url}: {error}")
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

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
