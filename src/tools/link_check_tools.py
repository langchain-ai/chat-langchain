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
ALLOWED_HOSTS = SOFT_404_DOMAINS | {"www.langchain.com"}
OUT_OF_SCOPE_ERROR = "Out of scope: only LangChain documentation hosts can be checked"
METADATA_IPS = {
    ipaddress.ip_address("100.100.100.200"),
    ipaddress.ip_address("169.254.169.254"),
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
    """Check if a URL uses an allowed documentation host."""
    try:
        result = urlparse(url)
        return result.scheme in ("http", "https") and result.hostname in ALLOWED_HOSTS
    except Exception:
        return False


def _is_safe_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if a resolved address is safe to contact."""
    return address not in METADATA_IPS and address.is_global


async def _validate_destination(url: str) -> None:
    """Validate the URL host and every address returned by DNS."""
    if not _is_valid_url(url):
        raise ValueError(OUT_OF_SCOPE_ERROR)

    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(OUT_OF_SCOPE_ERROR)
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError(OUT_OF_SCOPE_ERROR)

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = await asyncio.to_thread(
        socket.getaddrinfo,
        hostname,
        port,
        type=socket.SOCK_STREAM,
    )
    if not addresses or any(
        not _is_safe_ip(ipaddress.ip_address(address[4][0])) for address in addresses
    ):
        raise ValueError(OUT_OF_SCOPE_ERROR)


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
    # Check cache first
    if url in _cache:
        return _cache[url]

    if not _is_valid_url(url):
        result = LinkCheckResult(url=url, valid=False, error=OUT_OF_SCOPE_ERROR)
        _cache[url] = result
        return result

    try:
        await _validate_destination(url)
        needs_content_check = _needs_soft_404_check(url)

        if needs_content_check:
            current_url = url
            for redirect_count in range(MAX_REDIRECTS + 1):
                await _validate_destination(current_url)
                async with client.stream(
                    "GET",
                    current_url,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    status_code = response.status_code
                    location = response.headers.get("Location")
                    if 300 <= status_code < 400 and location:
                        if redirect_count == MAX_REDIRECTS:
                            result = LinkCheckResult(
                                url=url,
                                valid=False,
                                error="Too many redirects",
                            )
                            _cache[url] = result
                            return result
                        current_url = urljoin(current_url, location)
                        continue

                    final_url = current_url if current_url != url else None
                    is_valid = 200 <= status_code < 400
                    content = ""
                    if is_valid and status_code == 200:
                        async for chunk in response.aiter_text():
                            content += chunk
                            if len(content) >= CONTENT_CHECK_BYTES:
                                break

                    if is_valid and status_code == 200 and _is_soft_404(content):
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
                        status_code=status_code,
                        final_url=final_url,
                        error=None if is_valid else f"HTTP {status_code}",
                    )
                    break
        else:
            # Use HEAD for non-langchain domains (much faster)
            response = await client.head(url, timeout=timeout, follow_redirects=False)

            # Some servers don't support HEAD, fall back to GET
            if response.status_code == 405:
                response = await client.get(
                    url, timeout=timeout, follow_redirects=False
                )

            final_url = str(response.url) if str(response.url) != url else None
            is_valid = 200 <= response.status_code < 400

            result = LinkCheckResult(
                url=url,
                valid=is_valid,
                status_code=response.status_code,
                final_url=final_url,
                error=None if is_valid else f"HTTP {response.status_code}",
            )

        _cache[url] = result
        return result

    except ValueError as e:
        result = LinkCheckResult(url=url, valid=False, error=str(e))
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
