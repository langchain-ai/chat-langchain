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
HOST_NOT_PERMITTED = (
    "Host not permitted: check_links only validates official LangChain documentation URLs"
)
ALLOWED_HOSTS = {
    "docs.langchain.com",
    "support.langchain.com",
    "www.langchain.com",
    "langchain-ai.github.io",
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
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _validate_url(url: str) -> tuple[str, str] | str:
    """Validate a URL and return its hostname and resolved address."""
    if not _is_valid_url(url):
        return "Invalid URL format"

    parsed = urlparse(url)
    hostname = parsed.hostname
    if parsed.scheme != "https" or hostname is None or hostname.lower() not in ALLOWED_HOSTS:
        return HOST_NOT_PERMITTED

    try:
        addresses = {
            sockaddr[4][0]
            for sockaddr in socket.getaddrinfo(
                hostname,
                parsed.port or 443,
                type=socket.SOCK_STREAM,
            )
        }
    except (OSError, ValueError):
        return HOST_NOT_PERMITTED

    safe_addresses = []
    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError:
            return HOST_NOT_PERMITTED
        if (
            parsed_address.is_loopback
            or parsed_address.is_private
            or parsed_address.is_link_local
            or parsed_address.is_reserved
            or parsed_address.is_multicast
            or not parsed_address.is_global
        ):
            return HOST_NOT_PERMITTED
        safe_addresses.append(address)

    return (hostname, safe_addresses[0]) if safe_addresses else HOST_NOT_PERMITTED


class _PinnedHTTPTransport(httpx.AsyncHTTPTransport):
    """Connect to a previously validated address while preserving TLS SNI."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        address = request.extensions.pop("validated_address")
        original_url = request.url
        original_host = request.headers.get("host")
        request.url = request.url.copy_with(host=address)
        request.headers["host"] = original_host or original_url.host
        request.extensions["sni_hostname"] = original_url.host
        try:
            return await super().handle_async_request(request)
        finally:
            request.url = original_url
            if original_host is None:
                del request.headers["host"]
            else:
                request.headers["host"] = original_host


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
        result = LinkCheckResult(url=url, valid=False, error="Invalid URL format")
        _cache[url] = result
        return result

    validated = _validate_url(url)
    if isinstance(validated, str):
        result = LinkCheckResult(url=url, valid=False, error=validated)
        _cache[url] = result
        return result

    _, address = validated

    def _request_kwargs() -> dict[str, object]:
        return {
            "timeout": timeout,
            "follow_redirects": False,
        }

    def _redirect_error(response: httpx.Response) -> str | None:
        location = getattr(response, "headers", {}).get("location")
        if 300 <= response.status_code < 400 and location:
            redirect_url = str(httpx.URL(str(response.url)).join(location))
            redirect_validation = _validate_url(redirect_url)
            if isinstance(redirect_validation, str):
                return redirect_validation
        return None

    try:
        needs_content_check = _needs_soft_404_check(url)
        request_extensions = {"validated_address": address}

        if needs_content_check:
            # Stream response, only read first chunk for soft 404 detection
            async with client.stream("GET", url, extensions=request_extensions, **_request_kwargs()) as response:
                redirect_error = _redirect_error(response)
                if redirect_error:
                    result = LinkCheckResult(url=url, valid=False, error=redirect_error)
                    _cache[url] = result
                    return result
                final_url = str(response.url) if str(response.url) != url else None
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
                        _cache[url] = result
                        return result

                result = LinkCheckResult(
                    url=url, valid=is_valid, status_code=response.status_code,
                    final_url=final_url, error=None if is_valid else f"HTTP {response.status_code}",
                )
        else:
            # Use HEAD for non-langchain domains (much faster)
            response = await client.head(url, extensions=request_extensions, **_request_kwargs())

            # Some servers don't support HEAD, fall back to GET
            if response.status_code == 405:
                response = await client.get(url, extensions=request_extensions, **_request_kwargs())

            redirect_error = _redirect_error(response)
            if redirect_error:
                result = LinkCheckResult(url=url, valid=False, error=redirect_error)
                _cache[url] = result
                return result

            final_url = str(response.url) if str(response.url) != url else None
            is_valid = 200 <= response.status_code < 400

            result = LinkCheckResult(
                url=url, valid=is_valid, status_code=response.status_code,
                final_url=final_url, error=None if is_valid else f"HTTP {response.status_code}",
            )

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
        max_redirects=MAX_REDIRECTS,
        transport=_PinnedHTTPTransport(),
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
