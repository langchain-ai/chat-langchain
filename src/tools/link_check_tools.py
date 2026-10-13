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
ALLOWED_HOSTS = {"langchain.com", "langchain-ai.github.io", "github.com"}

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
            and (result.port is None or result.port > 0)
            and not any(character.isspace() or ord(character) < 32 for character in url)
            and httpx.URL(url).host == result.hostname
        )
    except Exception:
        return False


def _is_allowed_host(url: str) -> bool:
    """Check exact approved hosts and dot-delimited LangChain subdomains."""
    hostname = urlparse(url).hostname or ""
    return hostname in ALLOWED_HOSTS or hostname.endswith(".langchain.com")


class _LinkPolicyError(ValueError):
    """Represent a blocked link without exposing destination details."""


def _validate_destination(url: str) -> None:
    """Reject malformed URLs and hosts outside the citation allowlist."""
    if not _is_valid_url(url):
        raise _LinkPolicyError("Invalid URL format")
    if not _is_allowed_host(url):
        raise _LinkPolicyError(
            "Skipped host: only approved LangChain citation hosts are allowed"
        )


class _PublicOnlyTransport(httpx.AsyncHTTPTransport):
    """Connect only to validated public addresses while retaining the TLS hostname."""

    def __init__(self) -> None:
        """Disable proxies and cross-host reuse of IP-keyed TLS connections."""
        super().__init__(
            trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Validate DNS and pin the connection to a public address for each request."""
        _validate_destination(str(request.url))
        hostname = request.url.host
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        try:
            addresses = await asyncio.wait_for(
                asyncio.to_thread(
                    socket.getaddrinfo, hostname, port, type=socket.SOCK_STREAM
                ),
                timeout=request.extensions.get("timeout", {}).get("connect"),
            )
            if not addresses:
                raise ValueError("No addresses")
            for address in addresses:
                parsed_address = ipaddress.ip_address(address[4][0])
                if (
                    not parsed_address.is_global
                    or parsed_address.is_loopback
                    or parsed_address.is_private
                    or parsed_address.is_link_local
                    or parsed_address.is_reserved
                    or parsed_address.is_multicast
                    or parsed_address.is_unspecified
                ):
                    raise ValueError("Non-public address")
        except (OSError, ValueError, TimeoutError) as exc:
            raise _LinkPolicyError(
                "Skipped host: public destination could not be verified"
            ) from exc

        pinned_request = httpx.Request(
            request.method,
            request.url.copy_with(host=addresses[0][4][0]),
            headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": hostname},
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
    try:
        _validate_destination(url)
        if url in _cache:
            return _cache[url]

        current_url = url
        method = "GET" if _needs_soft_404_check(url) else "HEAD"
        redirects = 0
        while True:
            _validate_destination(current_url)
            async with client.stream(
                method, current_url, timeout=timeout, follow_redirects=False
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if location:
                        if redirects >= MAX_REDIRECTS:
                            raise httpx.TooManyRedirects("Too many redirects")
                        current_url = urljoin(current_url, location)
                        _validate_destination(current_url)
                        redirects += 1
                        if (
                            _needs_soft_404_check(current_url)
                            or response.status_code == 303
                        ):
                            method = "GET"
                        continue

                if method == "HEAD" and response.status_code == 405:
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

    except _LinkPolicyError as exc:
        result = LinkCheckResult(url=url, valid=False, error=str(exc))
    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError:
        result = LinkCheckResult(url=url, valid=False, error="Connection failed")
    except Exception:
        logger.warning("Error checking link", exc_info=True)
        result = LinkCheckResult(url=url, valid=False, error="Link check failed")

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        trust_env=False,
        transport=_PublicOnlyTransport(),
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
    """Validate intended documentation/support citations on langchain.com and its subdomains, langchain-ai.github.io, or github.com, not arbitrary user URLs."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
