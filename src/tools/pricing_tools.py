"""Tool for fetching live pricing information from langchain.com/pricing."""

import logging
import re
import threading
import time

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

PRICING_URL = "https://www.langchain.com/pricing"
TIMEOUT = 15.0
USER_AGENT = "LangChain-SupportAgent/1.0"

# In-process TTL cache. The pricing page changes a handful of times per year,
# so a 1-hour TTL is well within the freshness budget. Per-process means each
# replica refetches independently — fine, the page is cheap to fetch and
# refetching once an hour per replica is negligible load on langchain.com.
_CACHE_TTL_SECONDS = 3600
_cache_lock = threading.Lock()
_cached_text: str | None = None
_cached_at: float = 0.0


def _extract_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(
        r"<script\b[^>]*>[\s\S]*?</script\b[^>]*>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<style\b[^>]*>[\s\S]*?</style\b[^>]*>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def _fetch_pricing_uncached() -> str:
    """Fetch and parse the live pricing page. Raises on failure."""
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = await client.get(PRICING_URL, timeout=TIMEOUT)
        response.raise_for_status()
        return _extract_text(response.text)


@tool
async def fetch_langchain_pricing() -> str:
    """ALWAYS use this tool for ANY question about LangChain pricing, plans, or trace limits.

    DO NOT use docs search for pricing questions — it does not have current pricing data.

    Triggers: "how many traces", "plus plan", "developer plan", "enterprise plan",
    "how much does", "pricing", "cost", "seats", "quota", "pay-as-you-go", "fleet runs",
    "upgrade", "billing", "what plan", "which plan".

    Returns live pricing data directly from https://www.langchain.com/pricing.
    """
    global _cached_text, _cached_at

    # Fast path: return cached text if it's still fresh. The lock here is
    # only protecting the read of two related fields, not the network call.
    with _cache_lock:
        if (
            _cached_text is not None
            and (time.monotonic() - _cached_at) < _CACHE_TTL_SECONDS
        ):
            return _cached_text

    # Slow path: fetch outside the lock so concurrent requests don't serialize
    # on the network call. Multiple concurrent misses will all fetch and the
    # last writer wins, which is fine — the values are equivalent and the
    # extra fetches only happen at cache expiry.
    try:
        text = await _fetch_pricing_uncached()
    except httpx.TimeoutException:
        # On failure, fall back to stale cache if we have one — better than
        # telling the user "go check the website" when we have a 1-hour-old copy.
        with _cache_lock:
            if _cached_text is not None:
                logger.warning("Pricing fetch timed out, returning stale cached copy")
                return _cached_text
        return f"Error: Request to {PRICING_URL} timed out. Direct the user to {PRICING_URL} for current pricing."
    except httpx.HTTPStatusError as e:
        with _cache_lock:
            if _cached_text is not None:
                logger.warning(
                    f"Pricing fetch returned HTTP {e.response.status_code}, returning stale cached copy"
                )
                return _cached_text
        return f"Error: {PRICING_URL} returned HTTP {e.response.status_code}. Direct the user to {PRICING_URL} for current pricing."
    except Exception as e:
        logger.warning(f"Failed to fetch pricing page: {e}")
        with _cache_lock:
            if _cached_text is not None:
                return _cached_text
        return f"Error: Could not fetch pricing information. Direct the user to {PRICING_URL} for current pricing."

    with _cache_lock:
        _cached_text = text
        _cached_at = time.monotonic()
    return text
