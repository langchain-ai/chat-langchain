# Pylon Knowledge Base Tools
# Tools:
#   - search_support_articles
#   - get_support_article_content
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

# Pylon API configuration
PYLON_API_BASE_URL = "https://api.usepylon.com"


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

_articles_cache: Optional[List[Dict[str, Any]]] = None
_collections_cache: Optional[Dict[str, str]] = None


def _get_headers() -> Dict[str, str]:
    """Get API headers with authentication."""
    return {"Authorization": f"Bearer {_get_api_key()}", "Accept": "application/json"}


def _fetch_collections() -> Dict[str, str]:
    """Fetch collections from Pylon API and cache them.

    Returns:
        Mapping of collection names to collection IDs
    """
    global _collections_cache

    if _collections_cache is not None:
        return _collections_cache

    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/collections"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()

    collections_data = response.json().get("data", [])

    # Build mapping of collection names to IDs (only public collections)
    _collections_cache = {
        coll["title"]: coll["id"]
        for coll in collections_data
        if coll.get("visibility_config", {}).get("visibility") == "public"
    }

    return _collections_cache


def _fetch_all_articles() -> List[Dict[str, Any]]:
    """Fetch all articles from Pylon API and cache them.

    Follows pagination cursors until all pages are retrieved, with a safety
    cap of 10 pages (~1000 articles) to prevent infinite loops.
    """
    global _articles_cache

    if _articles_cache is not None:
        return _articles_cache

    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/articles"
    headers = _get_headers()

    all_articles: List[Dict[str, Any]] = []
    max_pages = 10
    pages_fetched = 0
    params: Dict[str, Any] = {}

    while pages_fetched < max_pages:
        response = requests.get(url, headers=headers, params=params)
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

    _articles_cache = all_articles
    return _articles_cache


# =============================================================================
# Identifier Resolution Helpers
# =============================================================================

_PLACEHOLDER_ARTICLE_IDS = {"", "dummy", "example", "<id>", "article_id", "id"}


def _normalize_collection_key(name: str) -> tuple:
    """Normalize a collection name to (prefix, frozenset of parenthetical tokens)."""
    lowered = name.strip().casefold()
    match = re.match(r"^(.*?)\s*\((.*)\)\s*$", lowered)
    if not match:
        return (lowered, frozenset())
    prefix = match.group(1).strip()
    tokens = frozenset(t for t in re.split(r"[\s,]+|\band\b", match.group(2)) if t)
    return (prefix, tokens)


def _resolve_collection_name(name: str, collection_map: Dict[str, str]) -> str | None:
    """Resolve a requested collection name to a real key in collection_map."""
    if name in collection_map:
        return name

    stripped = name.strip().casefold()
    for key in collection_map:
        if key.casefold() == stripped:
            return key

    target_prefix, target_tokens = _normalize_collection_key(name)
    for key in collection_map:
        if _normalize_collection_key(key) == (target_prefix, target_tokens):
            return key

    # Models often mangle the parenthetical part ("OSS (LangGraph and LangGraph)"),
    # so fall back to a subset match when it identifies exactly one collection.
    subset_matches = [
        key
        for key in collection_map
        if target_tokens
        and _normalize_collection_key(key)[0] == target_prefix
        and target_tokens <= _normalize_collection_key(key)[1]
    ]
    if len(subset_matches) == 1:
        return subset_matches[0]

    return None


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
                    Collection names are matched leniently: casing and the word
                    order inside parentheses do not matter. Any name that cannot
                    be resolved is reported in "unresolved_collections" while the
                    collections that did resolve are still searched.

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
        unresolved_collections: List[str] = []
        if collections.lower() != "all":
            # Parse requested collection names
            requested_collections = [c.strip() for c in collections.split(",")]

            # Get collection IDs for requested collections
            collection_ids = []
            for coll_name in requested_collections:
                resolved = _resolve_collection_name(coll_name, collection_map)
                if resolved is not None:
                    collection_ids.append(collection_map[resolved])
                else:
                    unresolved_collections.append(coll_name)

            # Only fail the whole call when nothing at all could be resolved;
            # otherwise a single bad name would discard every valid collection.
            if not collection_ids:
                return json.dumps(
                    {
                        "error": f"Collection(s) {', '.join(repr(c) for c in unresolved_collections)} not found. Available collections: {', '.join(collection_map.keys())}"
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
        if unresolved_collections:
            result["unresolved_collections"] = unresolved_collections
            result["available_collections"] = list(collection_map.keys())

        return json.dumps(result, indent=2)

    except ValueError as e:
        # API key not configured
        return json.dumps({"error": str(e)}, indent=2)
    except requests.exceptions.RequestException as e:
        # Network/API error
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        # Catch-all for unexpected errors
        return json.dumps({"error": f"Unexpected error: {str(e)}"}, indent=2)


@tool
def get_support_article_content(article_id: str) -> str:
    """Fetch the full HTML content of a specific Pylon support article.

    Uses cached articles from search_support_articles to avoid redundant API calls.
    Prefer the UUID "id" returned by search_support_articles; the
    support.langchain.com "{identifier}-{slug}" value from that article's "url"
    (or the full URL) is also accepted. Do not pass docs.langchain.com URLs or
    paths.

    Args:
        article_id: The article's UUID "id" from search_support_articles, or its
            support.langchain.com "{identifier}-{slug}" value or full URL

    Returns:
        Article content with only: id, title, url, collection, content
    """
    try:
        lookup = re.sub(
            r"^https?://support\.langchain\.com/articles/",
            "",
            (article_id or "").strip(),
        ).strip("/")
        if lookup.casefold() in _PLACEHOLDER_ARTICLE_IDS:
            return (
                f"Invalid article_id {article_id!r}. Call search_support_articles "
                "first and pass the 'id' value of the article you want."
            )

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
            identifier = str(article.get("identifier") or "")
            slug = str(article.get("slug") or "")
            candidates = {
                str(article.get("id") or ""),
                identifier,
                slug,
                f"{identifier}-{slug}" if identifier and slug else "",
            }
            candidates.discard("")
            if lookup in candidates:
                title = article.get("title", "Untitled")
                # Look up collection name by collection_id; fall back to default
                coll_id = article.get("collection_id")
                collection = collection_id_to_name.get(
                    coll_id, "Customer Support Knowledge Base"
                )

                # Construct support.langchain.com URL
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

        return (
            f"Article ID {article_id} not found in knowledge base. Call "
            "search_support_articles and use the 'id' value it returns."
        )

    except ValueError as e:
        # API key not configured
        return f"Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        # Network/API error
        return f"Error fetching article: {str(e)}"
    except Exception as e:
        # Catch-all for unexpected errors
        return f"Unexpected error: {str(e)}"


# Backwards-compatible Python import alias. The tool name exposed to the model is
# get_support_article_content, which avoids confusion with official docs pages.
get_article_content = get_support_article_content
