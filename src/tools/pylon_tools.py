# Pylon Knowledge Base Tools
# Tools:
#   - search_support_articles
#   - get_article_content
import logging
import os
import json
import requests
from typing import Any, Dict, List, Optional
from langchain.tools import tool
from dotenv import load_dotenv

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
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Accept": "application/json"
    }


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
    """Fetch all articles from Pylon API and cache them."""
    global _articles_cache

    if _articles_cache is not None:
        return _articles_cache

    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/articles"
    response = requests.get(url, headers=_get_headers())
    response.raise_for_status()

    _articles_cache = response.json().get("data", [])
    return _articles_cache


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
            return json.dumps({
                "collections": collections,
                "total": 0,
                "articles": [],
                "note": "No articles returned from API"
            }, indent=2)

        # Filter to only PUBLIC visibility articles with valid titles
        published_articles = []
        for article in articles:
            if (article.get("is_published", False)
                and article.get("title")
                and article.get("title") != "Untitled"
                and article.get("visibility_config", {}).get("visibility") == "public"
                and article.get("identifier")
                and article.get("slug")):

                # Construct support.langchain.com URL
                identifier = article.get("identifier")
                slug = article.get("slug")
                support_url = f"https://support.langchain.com/articles/{identifier}-{slug}"

                published_articles.append({
                    "id": article.get("id"),
                    "title": article.get("title", ""),
                    "url": support_url,
                    "collection_id": article.get("collection_id")  # Keep for filtering, will be set later
                })

        if not published_articles:
            return "No published articles available in the knowledge base."

        # Fetch collection map for naming
        try:
            collection_map = _fetch_collections()
        except Exception as e:
            return json.dumps({
                "error": f"Failed to fetch collections: {str(e)}"
            }, indent=2)

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
                        return json.dumps({
                            "error": f"Collection '{coll_name}' not found. Available collections: {', '.join(collection_map.keys())}"
                        }, indent=2)

            # Filter articles by collection_id
            filtered_articles = [
                article for article in published_articles
                if article.get("collection_id") in collection_ids
            ]

            published_articles = filtered_articles

        # Update collection names based on collection_id (for all articles)
        collection_id_to_name = {v: k for k, v in collection_map.items()}
        for article in published_articles:
            coll_id = article.get("collection_id")
            article["collection"] = collection_id_to_name.get(coll_id, "Unknown")

        if not published_articles:
            return json.dumps({
                "collections": collections,
                "total": 0,
                "articles": [],
                "note": "No articles found"
            }, indent=2)

        # Clean up collection_id from output (internal field)
        for article in published_articles:
            article.pop("collection_id", None)

        # Return structured JSON format
        result = {
            "collections": collections,
            "total": len(published_articles),
            "articles": published_articles,
            "note": "All articles listed are public and have content. Use IDs to fetch full content."
        }

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
def get_article_content(article_id: str) -> str:
    """Fetch the full HTML content of a specific support article.

    Uses cached articles from search_support_articles to avoid redundant API calls.

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

        # Find the article by ID
        for article in articles:
            if article.get("id") == article_id:
                # Extract collection from title (best guess based on keywords)
                title = article.get("title", "Untitled")
                collection = "Customer Support Knowledge Base"
                if "langgraph" in title.lower():
                    collection = "LangGraph"
                elif "langsmith" in title.lower():
                    collection = "LangSmith"
                elif "self" in title.lower() and "host" in title.lower():
                    collection = "Self Hosted"

                # Construct support.langchain.com URL
                identifier = article.get("identifier", "")
                slug = article.get("slug", "")
                if identifier and slug:
                    support_url = f"https://support.langchain.com/articles/{identifier}-{slug}"
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
        # Network/API error
        return f"Error fetching article: {str(e)}"
    except Exception as e:
        # Catch-all for unexpected errors
        return f"Unexpected error: {str(e)}"
