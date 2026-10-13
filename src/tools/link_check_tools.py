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
    "smith.langchain.com",
    "api.smith.langchain.com",
    "eu.smith.langchain.com",
    "github.com",
    "pypi.org",
    "www.npmjs.com",
}

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
    """Check whether a URL targets an allowed documentation host."""
    try:
        result = urlparse(url)
        host = (result.hostname or "").lower()
        return (
            result.scheme in ("http", "https")
            and (host in ALLOWED_HOSTS or host.endswith(".langchain.com"))
            and not result.username
            and not result.password
            and result.port in (None, 80, 443)
            and (host != "github.com" or result.path.startswith("/langchain-ai/"))
        )
    except ValueError:
        return False


async def _resolves_to_public_ip(host: str) -> bool:
    """Reject hosts with missing or non-public DNS addresses."""
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo, host, None, type=socket.SOCK_STREAM
        )
        if not addresses:
            return False
        for address in addresses:
            resolved_ip = ipaddress.ip_address(address[4][0])
            if isinstance(resolved_ip, ipaddress.IPv6Address):
                resolved_ip = resolved_ip.ipv4_mapped or resolved_ip
            if not resolved_ip.is_global or resolved_ip.is_multicast:
                return False
        return True
    except (OSError, ValueError):
        return False


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = (urlparse(url).hostname or "").lower()
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
    # Check cache first
    if url in _cache:
        return _cache[url]

    if not _is_valid_url(url):
        result = LinkCheckResult(
            url=url, valid=False, error="Not an allowed documentation host"
        )
        _cache[url] = result
        return result

    try:
        current_url = url
        timeout = min(DEFAULT_TIMEOUT, timeout)
        for redirect_count in range(MAX_REDIRECTS + 1):
            host = urlparse(current_url).hostname or ""
            if not _is_valid_url(current_url) or not await _resolves_to_public_ip(host):
                result = LinkCheckResult(
                    url=url,
                    valid=False,
                    error="Redirects outside allowed documentation hosts"
                    if redirect_count
                    else "Not an allowed documentation host",
                )
                break

            content = ""
            if _needs_soft_404_check(current_url):
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
                    response = await client.get(
                        current_url, timeout=timeout, follow_redirects=False
                    )

            if (
                response.status_code in (301, 302, 303, 307, 308)
                and "location" in response.headers
            ):
                if redirect_count == MAX_REDIRECTS:
                    result = LinkCheckResult(
                        url=url, valid=False, error="Too many redirects"
                    )
                    break
                current_url = urljoin(str(response.url), response.headers["location"])
                continue

            final_url = str(response.url) if str(response.url) != url else None
            is_valid = 200 <= response.status_code < 400
            error = None if is_valid else f"HTTP {response.status_code}"
            if response.status_code == 200 and _is_soft_404(content):
                is_valid = False
                error = "Soft 404: Page shows 'not found' content"
            result = LinkCheckResult(
                url=url,
                valid=is_valid,
                status_code=response.status_code,
                final_url=final_url,
                error=error,
            )
            break

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
        result = LinkCheckResult(
            url=url, valid=False, error="Unable to check documentation link"
        )

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        trust_env=False,
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
    """Check allowed documentation URLs with a timeout capped at 10 seconds."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, min(DEFAULT_TIMEOUT, timeout))
    return _format_results(results)
