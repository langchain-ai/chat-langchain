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

_ALLOWED_HOSTS = {
    "docs.langchain.com",
    "reference.langchain.com",
    "support.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "langchain.com",
    "www.langchain.com",
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


async def _is_allowed_url(url: str) -> bool:
    """Allow only HTTPS documentation hosts resolving exclusively to public IPs."""
    try:
        if "\\" in url or any(
            ord(character) <= 32 or ord(character) == 127 for character in url
        ):
            return False
        result = urlparse(url)
        host = result.hostname
        if (
            result.scheme != "https"
            or not host
            or result.username is not None
            or result.password is not None
            or result.port not in (None, 443)
            or result.netloc.endswith(":")
        ):
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return False
        if host.lower() not in _ALLOWED_HOSTS:
            return False
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM),
            timeout=DEFAULT_TIMEOUT,
        )
        return bool(addresses) and all(
            (address := ipaddress.ip_address(entry[4][0])).is_global
            and not address.is_multicast
            and not address.is_reserved
            for entry in addresses
        )
    except (ValueError, OSError, TimeoutError):
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
    if not await _is_allowed_url(url):
        return LinkCheckResult(
            url=url, valid=False, error="Not checked: host not allowed"
        )

    if url in _cache:
        return _cache[url]

    try:
        current_url = url
        method = "GET" if _needs_soft_404_check(url) else "HEAD"
        redirects = 0
        while True:
            async with client.stream(
                method, current_url, timeout=timeout, follow_redirects=False
            ) as response:
                if (
                    response.status_code in (301, 302, 303, 307, 308)
                    and "location" in response.headers
                ):
                    next_url = urljoin(current_url, response.headers["location"])
                    if not await _is_allowed_url(next_url):
                        return LinkCheckResult(
                            url=url, valid=False, error="Not checked: host not allowed"
                        )
                    if redirects >= MAX_REDIRECTS:
                        raise httpx.TooManyRedirects("Too many redirects")
                    redirects += 1
                    current_url = next_url
                    if _needs_soft_404_check(current_url):
                        method = "GET"
                    continue

                if method == "HEAD" and response.status_code == 405:
                    if not await _is_allowed_url(current_url):
                        return LinkCheckResult(
                            url=url, valid=False, error="Not checked: host not allowed"
                        )
                    method = "GET"
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
                if response.status_code == 200 and _needs_soft_404_check(current_url):
                    content = ""
                    async for chunk in response.aiter_text():
                        content += chunk
                        if len(content) >= CONTENT_CHECK_BYTES:
                            break
                    if _is_soft_404(content):
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
        trust_env=False,
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
async def check_links(urls: list[str]) -> str:
    """Check cited HTTPS documentation links on supported hosts before responding."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, DEFAULT_TIMEOUT)
    return _format_results(results)
