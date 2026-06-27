"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection
UNSAFE_DESTINATION_ERROR = "Destination is not permitted (private or internal address)"

# Domains known to have soft 404s (return 200 with "not found" content)
SOFT_404_DOMAINS = {
    "docs.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "support.langchain.com",
}

_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
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


def _ip_is_forbidden(ip: ipaddress._BaseAddress) -> bool:
    """Return True when an IP falls in a category we refuse to contact."""
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_safe_destination(url: str) -> bool:
    """Reject URLs that resolve to private/loopback/link-local/reserved addresses."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host or host in _BLOCKED_HOSTNAMES:
            return False
        try:
            ip = ipaddress.ip_address(host)
            return not _ip_is_forbidden(ip)
        except ValueError:
            pass
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if _ip_is_forbidden(ip):
                return False
        return True
    except Exception:
        return False


def _is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL format and points at a safe destination."""
    try:
        result = urlparse(url)
        if not all([result.scheme in ("http", "https"), result.netloc]):
            return False
        return _is_safe_destination(url)
    except Exception:
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

    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if any(phrase in title for phrase in ['not found', '404', 'page not found']):
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
        try:
            parsed = urlparse(url)
            scheme_ok = parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            scheme_ok = False
        error = UNSAFE_DESTINATION_ERROR if scheme_ok else "Invalid URL format"
        result = LinkCheckResult(url=url, valid=False, error=error)
        _cache[url] = result
        return result

    try:
        needs_content_check = _needs_soft_404_check(url)
        current_url = url
        redirects = 0

        while True:
            if needs_content_check:
                async with client.stream(
                    "GET", current_url, timeout=timeout, follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        next_url = response.headers.get("location")
                        redirects += 1
                        if not next_url:
                            result = LinkCheckResult(
                                url=url, valid=False, status_code=response.status_code,
                                error="Redirect missing Location header",
                            )
                            break
                        if redirects > MAX_REDIRECTS:
                            result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
                            break
                        next_url = str(httpx.URL(current_url).join(next_url))
                        if not _is_safe_destination(next_url):
                            result = LinkCheckResult(
                                url=url, valid=False, error=UNSAFE_DESTINATION_ERROR,
                            )
                            break
                        current_url = next_url
                        continue

                    final_url = current_url if current_url != url else None
                    is_valid = 200 <= response.status_code < 400

                    if is_valid and response.status_code == 200:
                        content = ""
                        async for chunk in response.aiter_text():
                            content += chunk
                            if len(content) >= CONTENT_CHECK_BYTES:
                                break

                        if _is_soft_404(content):
                            result = LinkCheckResult(
                                url=url, valid=False, status_code=200, final_url=final_url,
                                error="Soft 404: Page shows 'not found' content",
                            )
                            break

                    result = LinkCheckResult(
                        url=url, valid=is_valid, status_code=response.status_code,
                        final_url=final_url,
                        error=None if is_valid else f"HTTP {response.status_code}",
                    )
                    break
            else:
                response = await client.head(current_url, timeout=timeout, follow_redirects=False)
                if response.status_code == 405:
                    response = await client.get(current_url, timeout=timeout, follow_redirects=False)

                if response.is_redirect:
                    next_url = response.headers.get("location")
                    redirects += 1
                    if not next_url:
                        result = LinkCheckResult(
                            url=url, valid=False, status_code=response.status_code,
                            error="Redirect missing Location header",
                        )
                        break
                    if redirects > MAX_REDIRECTS:
                        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
                        break
                    next_url = str(httpx.URL(current_url).join(next_url))
                    if not _is_safe_destination(next_url):
                        result = LinkCheckResult(
                            url=url, valid=False, error=UNSAFE_DESTINATION_ERROR,
                        )
                        break
                    current_url = next_url
                    continue

                final_url = current_url if current_url != url else None
                is_valid = 200 <= response.status_code < 400
                result = LinkCheckResult(
                    url=url, valid=is_valid, status_code=response.status_code,
                    final_url=final_url,
                    error=None if is_valid else f"HTTP {response.status_code}",
                )
                break

        _cache[url] = result
        return result

    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError as e:
        result = LinkCheckResult(url=url, valid=False, error=f"Connection failed: {str(e)[:50]}")
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

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
