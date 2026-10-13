"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import posixpath
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
    "reference.langchain.com",
    "support.langchain.com",
    "www.langchain.com",
    "langchain.com",
    "academy.langchain.com",
    "github.com",
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


class _URLRejected(ValueError):
    """Signal that a destination must not be requested or cached."""


def _normalize_url(url: str) -> httpx.URL:
    """Normalize and enforce the URL destination policy."""
    try:
        if not url or any(
            character.isspace() or ord(character) < 32 for character in url
        ):
            raise ValueError
        normalized = httpx.URL(url).copy_with(fragment=None)
        if normalized.scheme not in ("http", "https") or not normalized.host:
            raise ValueError
    except (ValueError, httpx.InvalidURL) as error:
        raise _URLRejected("Invalid URL format") from error

    if (
        normalized.host not in ALLOWED_HOSTS
        or normalized.userinfo
        or normalized.port not in (None, 80 if normalized.scheme == "http" else 443)
    ):
        raise _URLRejected("Host not allowed")
    if normalized.host == "github.com" and (
        not posixpath.normpath(normalized.path).startswith("/langchain-ai/")
        or "\\" in normalized.path
        or "%" in normalized.path
    ):
        raise _URLRejected("Host not allowed")
    return normalized


async def _resolve_public_address(url: httpx.URL, timeout: float) -> str:
    """Resolve a hostname and reject any non-public answer."""
    try:
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                url.host,
                url.port or (80 if url.scheme == "http" else 443),
                type=socket.SOCK_STREAM,
            ),
            timeout=timeout,
        )
        if not addresses:
            raise ValueError
        for address in addresses:
            parsed = ipaddress.ip_address(address[4][0])
            if not parsed.is_global or parsed.is_reserved or parsed.is_multicast:
                raise ValueError
        return str(addresses[0][4][0])
    except (TimeoutError, OSError, ValueError) as error:
        raise _URLRejected("Host not allowed") from error


class _PinnedTransport(httpx.AsyncHTTPTransport):
    """Connect to the validated IP while retaining the original TLS identity."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send only to the address validated for this request."""
        address = request.extensions["validated_address"]
        extensions = {**request.extensions, "sni_hostname": request.url.host}
        pinned_request = httpx.Request(
            request.method,
            request.url.copy_with(host=address),
            headers=request.headers,
            stream=request.stream,
            extensions=extensions,
        )
        return await super().handle_async_request(pinned_request)


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
    redirects = 0
    try:
        current = _normalize_url(url)
        cache_key = str(current)
        while True:
            address = await _resolve_public_address(current, timeout)
            if redirects == 0 and cache_key in _cache:
                cached = _cache[cache_key]
                return LinkCheckResult(
                    url=url,
                    valid=cached.valid,
                    status_code=cached.status_code,
                    error=cached.error,
                    final_url=cached.final_url,
                )

            needs_content_check = _needs_soft_404_check(str(current))
            method = "GET" if needs_content_check else "HEAD"
            while True:
                async with client.stream(
                    method,
                    str(current),
                    timeout=timeout,
                    follow_redirects=False,
                    extensions={"validated_address": address},
                ) as response:
                    if response.status_code == 405 and method == "HEAD":
                        method = "GET"
                        continue
                    location = response.headers.get("location")
                    if response.status_code in (301, 302, 303, 307, 308) and location:
                        destination = _normalize_url(urljoin(str(current), location))
                        if redirects >= MAX_REDIRECTS:
                            raise httpx.TooManyRedirects("Too many redirects")
                        current = destination
                        redirects += 1
                        break

                    final_url = str(current) if str(current) != cache_key else None
                    is_valid = 200 <= response.status_code < 400
                    error = None if is_valid else f"HTTP {response.status_code}"
                    if needs_content_check and response.status_code == 200:
                        content = ""
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
                        final_url=final_url,
                        error=error,
                    )
                    if redirects == 0:
                        _cache[cache_key] = result
                    return result
    except _URLRejected as error:
        return LinkCheckResult(url=url, valid=False, error=str(error))
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

    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        trust_env=False,
        transport=_PinnedTransport(
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ),
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
    """Validate approved LangChain documentation URLs you plan to cite, not reachability probes."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
