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

ALLOWED_HOSTS = {
    "docs.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "support.langchain.com",
    "reference.langchain.com",
    "langchain-ai.github.io",
    "github.com",
}
HOST_NOT_ALLOWED = "Host not allowed for link checking"

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
    """Check whether an HTTP URL has an explicitly allowed hostname."""
    try:
        result = urlparse(url)
        return (
            result.scheme in ("http", "https")
            and result.hostname in ALLOWED_HOSTS
            and (result.port is None or 0 < result.port <= 65535)
        )
    except Exception:
        return False


async def _has_public_addresses(url: str, timeout: float) -> bool:
    """Resolve an allowed URL and reject any non-public destination address."""
    if not _is_valid_url(url):
        return False
    try:
        parsed = urlparse(url)
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ),
            timeout=timeout,
        )
        for address in addresses:
            resolved = ipaddress.ip_address(address[4][0])
            if (
                resolved.is_private
                or resolved.is_loopback
                or resolved.is_link_local
                or resolved.is_reserved
                or resolved.is_multicast
                or resolved.is_unspecified
                or not resolved.is_global
            ):
                return False
        return bool(addresses)
    except (OSError, ValueError, TimeoutError):
        return False


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = urlparse(url).hostname
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
    timeout = min(DEFAULT_TIMEOUT, timeout)
    current_url = url
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            if not await _has_public_addresses(current_url, timeout):
                return LinkCheckResult(url=url, valid=False, error=HOST_NOT_ALLOWED)
            if redirect_count == 0 and url in _cache:
                return _cache[url]

            needs_content_check = _needs_soft_404_check(current_url)
            content = ""
            if needs_content_check:
                async with client.stream(
                    "GET", current_url, timeout=timeout, follow_redirects=False
                ) as response:
                    if response.status_code == 200:
                        async for chunk in response.aiter_text():
                            content += chunk
                            if len(content) >= CONTENT_CHECK_BYTES:
                                break
            else:
                response = await client.head(
                    current_url, timeout=timeout, follow_redirects=False
                )
                if response.status_code == 405:
                    if not await _has_public_addresses(current_url, timeout):
                        return LinkCheckResult(
                            url=url, valid=False, error=HOST_NOT_ALLOWED
                        )
                    response = await client.get(
                        current_url, timeout=timeout, follow_redirects=False
                    )

            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if location:
                    try:
                        current_url = urljoin(current_url, location)
                    except ValueError:
                        return LinkCheckResult(
                            url=url, valid=False, error=HOST_NOT_ALLOWED
                        )
                    if not _is_valid_url(current_url):
                        return LinkCheckResult(
                            url=url, valid=False, error=HOST_NOT_ALLOWED
                        )
                    if redirect_count == MAX_REDIRECTS:
                        raise httpx.TooManyRedirects("Too many redirects")
                    continue

            final_url = current_url if current_url != url else None
            is_valid = 200 <= response.status_code < 300
            result = LinkCheckResult(
                url=url,
                valid=is_valid,
                status_code=response.status_code,
                final_url=final_url,
                error=None if is_valid else f"HTTP {response.status_code}",
            )
            if (
                needs_content_check
                and response.status_code == 200
                and _is_soft_404(content)
            ):
                result.valid = False
                result.error = "Soft 404: Page shows 'not found' content"
            _cache[url] = result
            return result

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

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
    ) as client:
        tasks = [_check_single_url(client, url, timeout) for url in urls]
        return list(await asyncio.gather(*tasks))


def _format_results(results: list[LinkCheckResult]) -> str:
    """Format check results into readable output."""
    if not results:
        return "No URLs to check."

    valid = [r for r in results if r.valid]
    rejected = [r for r in results if r.error == HOST_NOT_ALLOWED]
    invalid = [r for r in results if not r.valid and r.error != HOST_NOT_ALLOWED]

    lines = [
        f"Link Check Results: {len(valid)}/{len(valid) + len(invalid)} valid\n"
        if valid or invalid
        else "Link Check Results: no URLs checked\n"
    ]

    if rejected:
        lines.append("Rejected links (not checked):")
        lines.extend(f"  - {r.url}: {HOST_NOT_ALLOWED}" for r in rejected)
        lines.append("")

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
    """Check allowed LangChain docs and GitHub URLs with a timeout capped at 10 seconds."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, min(DEFAULT_TIMEOUT, timeout))
    return _format_results(results)
