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
        result = urlparse(url)
        return bool(
            result.scheme in ("http", "https")
            and result.hostname
            and result.username is None
            and result.password is None
            and not result.netloc.endswith(":")
            and (result.port is None or result.port > 0)
            and not re.search(r"[\s\x00-\x1f\x7f\\]", url)
            and "%" not in result.hostname
        )
    except ValueError:
        return False


def _public_address(address: str) -> str | None:
    """Return a normalized address only if it is publicly routable."""
    parsed = ipaddress.ip_address(address)
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
        parsed = parsed.ipv4_mapped
    if (
        not parsed.is_global
        or parsed.is_reserved
        or parsed.is_multicast
        or parsed.is_unspecified
        or parsed.is_loopback
        or parsed.is_link_local
    ):
        return None
    return str(parsed)


async def _validated_destination(
    url: str, timeout: float
) -> tuple[httpx.URL, str] | None:
    """Resolve a URL and pin it to an exclusively public destination."""
    if not _is_valid_url(url):
        return None
    try:
        parsed = httpx.URL(url)
        hostname = parsed.raw_host.decode("ascii").rstrip(".").lower()
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            labels = hostname.split(".")
            if (
                len(labels) < 2
                or len(hostname) > 253
                or any(
                    not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                    for label in labels
                )
                or labels[-1]
                in {
                    "localhost",
                    "local",
                    "localdomain",
                    "internal",
                    "lan",
                    "home",
                    "svc",
                    "test",
                    "invalid",
                    "onion",
                }
            ):
                return None
            records = await asyncio.wait_for(
                asyncio.get_running_loop().getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                ),
                timeout=timeout,
            )
            addresses = [_public_address(str(record[4][0])) for record in records]
        else:
            addresses = [_public_address(hostname)]
        if not addresses or any(address is None for address in addresses):
            return None
        address = addresses[0]
        assert address is not None
        return parsed.copy_with(host=address), hostname
    except (ValueError, httpx.InvalidURL, OSError, TimeoutError):
        return None


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
    current_url = url
    try:
        for redirects in range(MAX_REDIRECTS + 1):
            destination = await _validated_destination(current_url, timeout)
            if destination is None:
                return LinkCheckResult(url=url, valid=False, error="URL not allowed")
            if redirects == 0 and url in _cache and not _cache[url].final_url:
                return _cache[url]
            pinned_url, hostname = destination
            headers = {"Host": httpx.URL(current_url).netloc.decode("ascii")}
            extensions = {"sni_hostname": hostname}
            needs_content_check = _needs_soft_404_check(current_url)
            method = "GET" if needs_content_check else "HEAD"
            content = ""
            while True:
                async with client.stream(
                    method,
                    pinned_url,
                    headers=headers,
                    extensions=extensions,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if method == "HEAD" and response.status_code == 405:
                        method = "GET"
                        continue
                    if response.has_redirect_location:
                        if redirects == MAX_REDIRECTS:
                            raise httpx.TooManyRedirects("Too many redirects")
                        try:
                            current_url = str(
                                httpx.URL(current_url).join(
                                    response.headers["location"]
                                )
                            )
                        except (ValueError, httpx.InvalidURL):
                            return LinkCheckResult(
                                url=url, valid=False, error="URL not allowed"
                            )
                        break
                    is_valid = 200 <= response.status_code < 400
                    error = None if is_valid else f"HTTP {response.status_code}"
                    if needs_content_check and response.status_code == 200:
                        async for chunk in response.aiter_text():
                            content += chunk
                            if len(content) >= CONTENT_CHECK_BYTES:
                                break
                        if _is_soft_404(content):
                            is_valid = False
                            error = "Soft 404: Page shows 'not found' content"
                    result = LinkCheckResult(
                        url=url,
                        valid=is_valid,
                        status_code=response.status_code,
                        final_url=current_url if current_url != url else None,
                        error=error,
                    )
                    if redirects == 0:
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

    if redirects == 0:
        _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
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
