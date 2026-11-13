# Pylon Knowledge Base Tools
# Tools:
#   - search_support_articles
#   - get_support_article_content
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

# Pylon API configuration
PYLON_API_BASE_URL = "https://api.usepylon.com"
PYLON_CACHE_TTL_SECONDS = float(os.getenv("PYLON_CACHE_TTL_SECONDS", "900"))
PYLON_REQUEST_TIMEOUT_SECONDS = 30


def _get_kb_id() -> str:
    """Get knowledge base ID from environment."""
    kb_id = os.getenv("PYLON_KB_ID")
    if not kb_id:
        raise ValueError("PYLON_KB_ID not configured in .env")
    return kb_id


def _get_api_key() -> str:
    """Get Pylon API key from environment."""
    api_key = os.getenv("PYLON_API_KEY")
    if not api_key:
        raise ValueError("PYLON_API_KEY not configured in .env")
    return api_key


# =============================================================================
# Cache & API Helpers
# =============================================================================


class _TTLCache:
    """Process-wide cache that refetches after PYLON_CACHE_TTL_SECONDS."""

    def __init__(self) -> None:
        self.value: Any = None
        self.fetched_at = 0.0
        self.lock = threading.Lock()

    def _fresh(self) -> bool:
        return (
            self.value is not None
            and time.monotonic() - self.fetched_at < PYLON_CACHE_TTL_SECONDS
        )

    def get(self, loader: Callable[[], Any]) -> Any:
        if self._fresh():
            return self.value
        if not self.lock.acquire(blocking=self.value is None):
            return self.value
        try:
            if self._fresh():
                return self.value
            try:
                self.value = loader()
            except Exception:
                if self.value is None:
                    raise
                logger.warning(
                    "Pylon refresh failed, serving stale cache", exc_info=True
                )
            self.fetched_at = time.monotonic()
            return self.value
        finally:
            self.lock.release()

    def clear(self) -> None:
        self.value = None
        self.fetched_at = 0.0


_articles_cache = _TTLCache()
_collections_cache = _TTLCache()


def _get_headers() -> Dict[str, str]:
    """Get API headers with authentication."""
    return {"Authorization": f"Bearer {_get_api_key()}", "Accept": "application/json"}


def _fetch_collections() -> Dict[str, str]:
    """Fetch collections from Pylon API, cached for PYLON_CACHE_TTL_SECONDS.

    Returns:
        Mapping of collection names to collection IDs
    """
    return _collections_cache.get(_load_collections)


def _load_collections() -> Dict[str, str]:
    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/collections"
    response = requests.get(
        url, headers=_get_headers(), timeout=PYLON_REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()

    collections_data = response.json().get("data", [])

    # Build mapping of collection names to IDs (only public collections)
    return {
        coll["title"]: coll["id"]
        for coll in collections_data
        if coll.get("visibility_config", {}).get("visibility") == "public"
    }


def _fetch_all_articles() -> List[Dict[str, Any]]:
    """Fetch all articles from Pylon API, cached for PYLON_CACHE_TTL_SECONDS.

    Follows pagination cursors until all pages are retrieved, with a safety
    cap of 10 pages (~1000 articles) to prevent infinite loops.
    """
    return _articles_cache.get(_load_all_articles)


def _load_all_articles() -> List[Dict[str, Any]]:
    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/articles"
    headers = _get_headers()

    all_articles: List[Dict[str, Any]] = []
    max_pages = 10
    pages_fetched = 0
    params: Dict[str, Any] = {}

    while pages_fetched < max_pages:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=PYLON_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()

        page_data = body.get("data", [])
        all_articles.extend(page_data)
        pages_fetched += 1

        # Resolve next-page cursor from common Pylon/REST pagination shapes
        next_cursor = (
            body.get("next")
            or body.get("meta", {}).get("next")
            or body.get("links", {}).get("next")
            or body.get("pagination", {}).get("cursor")
        )

        if not next_cursor:
            break

        params = {"cursor": next_cursor}

    return all_articles


KB_UNAVAILABLE_NOTE = (
    "The LangChain support knowledge base is temporarily unavailable. Do not "
    "conclude that no support article exists, and do not call support article "
    "tools again in this turn. Answer from the official documentation. This is an "
    "internal backend issue: do not mention the support knowledge base, support "
    "articles, or this outage to the user."
)


def _kb_unavailable(error: Exception) -> str:
    """JSON the model can act on when the Pylon API is unreachable or rejects us."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    return json.dumps(
        {
            "support_kb_unavailable": True,
            "status": status,
            "error": str(error),
            "note": KB_UNAVAILABLE_NOTE,
        },
        indent=2,
    )


# =============================================================================
# LangChain Tools
# =============================================================================


@tool
def search_support_articles(collections: str = "all") -> str:
    """Get LangChain support article titles from Pylon KB, filtered by collection(s).

    Returns article titles in structured JSON format so the LLM can decide which ones to fetch.

    Args:
        collections: Comma-separated list of collection names to filter by.
                    Available collections:
                    - "General" - General administration and management topics
                    - "OSS (LangChain and LangGraph)" - Open source libraries for LangChain and LangGraph
                    - "LangSmith Observability" - Tracing, stats, and observability of agents
                    - "LangSmith Evaluation" - Datasets, evaluations, and prompts
                    - "LangSmith Deployment" - Graph runtime and deployments (formerly LangGraph Platform)
                    - "SDKs and APIs" - All things across SDKs and APIs
                    - "LangSmith Studio" - Visualizing and debugging agents (formerly LangGraph Studio)
                    - "Self Hosted" - Self-hosted LangSmith including deployments
                    - "Troubleshooting" - Broad domain issue triage and resolution
                    - "Security" - Code scans, key management, and security topics

                    Use "all" to search all collections (default)
                    Example: "LangSmith Deployment,LangSmith Observability" to get articles about both

    Returns:
        JSON string with structure: {"collections": "...", "total": N, "articles": [...]}
    """
    try:
        # Fetch and cache all articles (includes content)
        articles = _fetch_all_articles()

        # Handle None or empty response
        if articles is None or not articles:
            return json.dumps(
                {
                    "collections": collections,
                    "total": 0,
                    "articles": [],
                    "note": "No articles returned from API",
                },
                indent=2,
            )

        # Filter to only PUBLIC visibility articles with valid titles
        published_articles = []
        for article in articles:
            if (
                article.get("is_published", False)
                and article.get("title")
                and article.get("title") != "Untitled"
                and article.get("visibility_config", {}).get("visibility") == "public"
                and article.get("identifier")
                and article.get("slug")
            ):
                # Construct support.langchain.com URL
                identifier = article.get("identifier")
                slug = article.get("slug")
                support_url = (
                    f"https://support.langchain.com/articles/{identifier}-{slug}"
                )

                published_articles.append(
                    {
                        "id": article.get("id"),
                        "title": article.get("title", ""),
                        "url": support_url,
                        "collection_id": article.get(
                            "collection_id"
                        ),  # Keep for filtering, will be set later
                    }
                )

        if not published_articles:
            return "No published articles available in the knowledge base."

        # Fetch collection map for naming
        try:
            collection_map = _fetch_collections()
        except Exception as e:
            return json.dumps(
                {"error": f"Failed to fetch collections: {str(e)}"}, indent=2
            )

        # Filter by collection ID if specified
        if collections.lower() != "all":
            # Parse requested collection names
            requested_collections = [c.strip() for c in collections.split(",")]

            # Get collection IDs for requested collections
            collection_ids = []
            for coll_name in requested_collections:
                if coll_name in collection_map:
                    collection_ids.append(collection_map[coll_name])
                else:
                    # Try case-insensitive match
                    matched = False
                    for key in collection_map.keys():
                        if key.lower() == coll_name.lower():
                            collection_ids.append(collection_map[key])
                            matched = True
                            break
                    if not matched:
                        return json.dumps(
                            {
                                "error": f"Collection '{coll_name}' not found. Available collections: {', '.join(collection_map.keys())}"
                            },
                            indent=2,
                        )

            # Filter articles by collection_id
            filtered_articles = [
                article
                for article in published_articles
                if article.get("collection_id") in collection_ids
            ]

            published_articles = filtered_articles

        # Update collection names based on collection_id (for all articles)
        collection_id_to_name = {v: k for k, v in collection_map.items()}
        for article in published_articles:
            coll_id = article.get("collection_id")
            article["collection"] = collection_id_to_name.get(coll_id, "Unknown")

        if not published_articles:
            return json.dumps(
                {
                    "collections": collections,
                    "total": 0,
                    "articles": [],
                    "note": "No articles found",
                },
                indent=2,
            )

        # Clean up collection_id from output (internal field)
        for article in published_articles:
            article.pop("collection_id", None)

        # Return structured JSON format
        result = {
            "collections": collections,
            "total": len(published_articles),
            "articles": published_articles,
            "note": "All articles listed are public and have content. Use IDs to fetch full content.",
        }

        return json.dumps(result, indent=2)

    except ValueError as e:
        # API key not configured
        return json.dumps({"error": str(e)}, indent=2)
    except requests.exceptions.RequestException as e:
        # Network/API error (for example a rejected or expired API key)
        return _kb_unavailable(e)
    except Exception as e:
        # Catch-all for unexpected errors
        return json.dumps({"error": f"Unexpected error: {str(e)}"}, indent=2)


@tool
def get_support_article_content(article_id: str) -> str:
    """Fetch the full HTML content of a specific Pylon support article.

    Uses cached articles from search_support_articles to avoid redundant API calls.
    This only accepts article IDs returned by search_support_articles; do not pass
    docs.langchain.com URLs or paths.

    Args:
        article_id: The article ID from search_support_articles

    Returns:
        Article content with only: id, title, url, collection, content
    """
    try:
        # Use cached articles (already fetched by search_support_articles)
        articles = _fetch_all_articles()

        # Handle None or empty response
        if articles is None or not articles:
            return "Error: No articles available from API. Check PYLON_API_KEY configuration."

        # Build reverse mapping: collection_id -> collection_name
        try:
            collection_map = _fetch_collections()
            collection_id_to_name = {v: k for k, v in collection_map.items()}
        except Exception:
            collection_id_to_name = {}

        # Find the article by ID
        for article in articles:
            if article.get("id") == article_id:
                title = article.get("title", "Untitled")
                # Look up collection name by collection_id; fall back to default
                coll_id = article.get("collection_id")
                collection = collection_id_to_name.get(
                    coll_id, "Customer Support Knowledge Base"
                )

                # Construct support.langchain.com URL
                identifier = article.get("identifier", "")
                slug = article.get("slug", "")
                if identifier and slug:
                    support_url = (
                        f"https://support.langchain.com/articles/{identifier}-{slug}"
                    )
                else:
                    support_url = "URL not available"

                # Only return id, title, url, collection, content
                return f"""ID: {article.get("id")}
Title: {title}
URL: {support_url}
Collection: {collection}

Content:
{article.get("current_published_content_html", "No content available")[:5000]}"""

        return f"Article ID {article_id} not found in knowledge base."

    except ValueError as e:
        # API key not configured
        return f"Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        # Network/API error (for example a rejected or expired API key)
        return _kb_unavailable(e)
    except Exception as e:
        # Catch-all for unexpected errors
        return f"Unexpected error: {str(e)}"


# Backwards-compatible Python import alias. The tool name exposed to the model is
# get_support_article_content, which avoids confusion with official docs pages.
get_article_content = get_support_article_content
