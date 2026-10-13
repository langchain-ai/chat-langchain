"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import posixpath
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
ALLOWED_HOST_SUFFIXES = ("langchain.com", "langsmith.com")
DISALLOWED_HOST_ERROR = "Not checked: only LangChain documentation hosts are allowed"
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
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def _is_allowed_host(url: str) -> bool:
    """Allow only LangChain citation hosts and the LangChain GitHub organization."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not _is_valid_url(url) or parsed.username or parsed.password:
            return False
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            return False
        if hostname == "github.com":
            path = unquote(parsed.path)
            return (
                "\\" not in path
                and path.startswith("/langchain-ai/")
                and (posixpath.normpath(path) + "/").startswith("/langchain-ai/")
            )
        return any(
            hostname == suffix or hostname.endswith("." + suffix)
            for suffix in ALLOWED_HOST_SUFFIXES
        )
    except ValueError:
        return False


async def _has_public_address(url: str, timeout: float) -> bool:
    """Reject hosts with any non-public resolved address."""
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
        ip = ipaddress.ip_address(address[4][0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return bool(addresses)


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
    # Check cache first
    if url in _cache:
        return _cache[url]

    if not _is_valid_url(url):
        result = LinkCheckResult(url=url, valid=False, error="Invalid URL format")
        _cache[url] = result
        return result

    if not _is_allowed_host(url):
        result = LinkCheckResult(url=url, valid=False, error=DISALLOWED_HOST_ERROR)
        _cache[url] = result
        return result

    try:
        current_url = str(httpx.URL(url))
        for redirect_count in range(MAX_REDIRECTS + 1):
            if not _is_allowed_host(current_url):
                result = LinkCheckResult(
                    url=url, valid=False, error=DISALLOWED_HOST_ERROR
                )
                break
            if not await _has_public_address(current_url, timeout):
                result = LinkCheckResult(
                    url=url,
                    valid=False,
                    error="Not checked: host resolves to a non-public address",
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

            if response.status_code in (
                301,
                302,
                303,
                307,
                308,
            ) and response.headers.get("location"):
                next_url = str(
                    httpx.URL(urljoin(current_url, response.headers["location"]))
                )
                if not _is_allowed_host(next_url):
                    result = LinkCheckResult(
                        url=url, valid=False, error=DISALLOWED_HOST_ERROR
                    )
                    break
                if redirect_count == MAX_REDIRECTS:
                    result = LinkCheckResult(
                        url=url, valid=False, error="Too many redirects"
                    )
                    break
                current_url = next_url
                continue

            final_url = current_url if current_url != url else None
            is_valid = 200 <= response.status_code < 300
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

    except (httpx.TimeoutException, TimeoutError):
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError:
        result = LinkCheckResult(url=url, valid=False, error="Connection failed")
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        result = LinkCheckResult(url=url, valid=False, error="Unable to check URL")

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
        safe_errors = {
            "Invalid URL format",
            DISALLOWED_HOST_ERROR,
            "Not checked: host resolves to a non-public address",
            "Request timed out",
            "Too many redirects",
            "Connection failed",
            "Soft 404: Page shows 'not found' content",
        }
        for r in invalid:
            error = (
                r.error
                if r.error in safe_errors or re.fullmatch(r"HTTP \d{3}", r.error or "")
                else "Unable to check URL"
            )
            lines.append(f"  - {r.url}: {error}")
        lines.append("")

    if valid:
        lines.append("Valid links:")
        for r in valid:
            suffix = (
                f" (→ {r.final_url})"
                if r.final_url and _is_allowed_host(r.final_url)
                else ""
            )
            lines.append(f"  - {r.url}{suffix}")

    return "\n".join(lines)


@tool
async def check_links(urls: list[str], timeout: float = DEFAULT_TIMEOUT) -> str:
    """Validate LangChain/LangSmith citation URLs and github.com/langchain-ai/ links."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
