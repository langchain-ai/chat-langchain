"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from langchain.tools import tool

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
MAX_CACHE_SIZE = 512
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection

ALLOWED_HOSTS = {
    "docs.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "support.langchain.com",
    "www.langchain.com",
    "langchain.com",
    "reference.langchain.com",
    "github.com",
    "pypi.org",
    "www.npmjs.com",
    "npmjs.com",
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
    """Check if a string is a valid URL format."""
    try:
        result = httpx.URL(url)
        return result.scheme in ("http", "https") and bool(result.host)
    except Exception:
        return False


def _url_error(url: str) -> str | None:
    """Reject malformed URLs and hosts outside the documentation allowlist."""
    if not _is_valid_url(url):
        return "Invalid URL format"
    hostname = httpx.URL(url).raw_host.decode("ascii").lower().rstrip(".")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        if len(hostname) <= 253 and all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in hostname.split(".")
        ):
            if hostname in ALLOWED_HOSTS or hostname.endswith(".langchain.com"):
                return None
    return "Host not allowed"


class _SafeTransport(httpx.AsyncBaseTransport):
    """Pin each connection to a public address while retaining the TLS hostname."""

    def __init__(self) -> None:
        # IP-keyed pooling must not reuse TLS sessions across different hostnames.
        self._transport = httpx.AsyncHTTPTransport(
            trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Validate DNS results and connect only to an approved numeric address."""
        error = _url_error(str(request.url))
        if error:
            raise ValueError(error)
        hostname = request.url.raw_host.decode("ascii")
        timeout = _clamp_timeout(request.extensions.get("timeout", {}).get("connect"))
        records = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                hostname,
                request.url.port or (443 if request.url.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ),
            timeout=timeout,
        )
        addresses = [ipaddress.ip_address(record[4][0]) for record in records]
        if not addresses or any(
            not address.is_global
            or address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
            for address in addresses
        ):
            raise ValueError("unreachable")
        headers = request.headers.copy()
        headers["Host"] = request.url.netloc.decode("ascii")
        pinned_request = httpx.Request(
            request.method,
            request.url.copy_with(host=str(addresses[0])),
            headers=headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": hostname},
        )
        return await self._transport.handle_async_request(pinned_request)

    async def aclose(self) -> None:
        """Close the underlying transport."""
        await self._transport.aclose()


def _clamp_timeout(timeout: float | None) -> float:
    """Limit network timeouts to one through ten seconds."""
    return max(
        1.0, min(DEFAULT_TIMEOUT, timeout if timeout is not None else DEFAULT_TIMEOUT)
    )


def _cache_result(result: LinkCheckResult) -> LinkCheckResult:
    """Cache a result without allowing unbounded key growth."""
    if result.url not in _cache and len(_cache) >= MAX_CACHE_SIZE:
        del _cache[next(iter(_cache))]
    _cache[result.url] = result
    return result


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = (urlparse(url).hostname or "").lower().rstrip(".")
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

    try:
        timeout = _clamp_timeout(timeout)
        current_url = url
        method = "GET" if _needs_soft_404_check(url) else "HEAD"
        redirects = 0
        while True:
            error = _url_error(current_url)
            if error:
                return _cache_result(LinkCheckResult(url=url, valid=False, error=error))
            async with client.stream(
                method, current_url, timeout=timeout, follow_redirects=False
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        return _cache_result(
                            LinkCheckResult(
                                url=url,
                                valid=False,
                                status_code=response.status_code,
                                error="Invalid redirect",
                            )
                        )
                    if redirects >= MAX_REDIRECTS:
                        return _cache_result(
                            LinkCheckResult(
                                url=url,
                                valid=False,
                                error="Too many redirects",
                            )
                        )
                    current_url = urljoin(current_url, location)
                    redirects += 1
                    if _needs_soft_404_check(current_url):
                        method = "GET"
                    continue
                if response.status_code == 405 and method == "HEAD":
                    method = "GET"
                    continue
                final_url = current_url if current_url != url else None
                is_valid = 200 <= response.status_code < 400
                if response.status_code == 200 and _needs_soft_404_check(current_url):
                    content = ""
                    async for chunk in response.aiter_text():
                        content += chunk
                        if len(content) >= CONTENT_CHECK_BYTES:
                            break

                    if _is_soft_404(content):
                        return _cache_result(
                            LinkCheckResult(
                                url=url,
                                valid=False,
                                status_code=200,
                                final_url=final_url,
                                error="Soft 404: Page shows 'not found' content",
                            )
                        )

                return _cache_result(
                    LinkCheckResult(
                        url=url,
                        valid=is_valid,
                        status_code=response.status_code,
                        final_url=final_url,
                        error=None if is_valid else f"HTTP {response.status_code}",
                    )
                )

    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except Exception:
        result = LinkCheckResult(url=url, valid=False, error="unreachable")

    return _cache_result(result)


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        transport=_SafeTransport(),
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
async def check_links(urls: list[str], timeout: float = DEFAULT_TIMEOUT) -> str:
    """Validate retrieved documentation URLs for citation with a 1–10 second timeout."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, _clamp_timeout(timeout))
    return _format_results(results)
