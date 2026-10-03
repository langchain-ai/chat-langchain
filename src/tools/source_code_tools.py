"""Read-only source lookup tools for public LangChain repositories."""

import base64
import json
import os
from collections.abc import Mapping
from typing import Any

import httpx
from langchain.tools import tool

GITHUB_API_URL = "https://api.github.com"
GITHUB_ORGANIZATION = "langchain-ai"
ALLOWED_REPOSITORIES = frozenset(
    {
        "langchain",
        "langgraph",
        "deepagents",
        "langchainjs",
        "langgraphjs",
        "deepagentsjs",
    }
)


def _validate_repository(repository: str) -> str:
    """Validate and return an allowlisted repository name."""
    if repository not in ALLOWED_REPOSITORIES:
        allowed = ", ".join(sorted(ALLOWED_REPOSITORIES))
        raise ValueError(f"Repository must be one of: {allowed}")
    return repository


def _validate_path(path: str) -> str:
    """Validate and return a repository-relative file path."""
    normalized_path = path.strip().lstrip("/")
    if (
        not normalized_path
        or normalized_path == "."
        or ".." in normalized_path.split("/")
    ):
        raise ValueError("Path must be a repository-relative file path")
    return normalized_path


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "chat-langchain-source-lookup",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _github_get(
    path: str, params: Mapping[str, str] | None = None
) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=GITHUB_API_URL, headers=_headers()) as client:
        response = await client.get(path, params=params, timeout=15.0)
        response.raise_for_status()
        return response.json()


@tool
async def search_langchain_source(repository: str, query: str, path: str = "") -> str:
    """Search source code in an allowlisted public langchain-ai repository."""
    try:
        repository = _validate_repository(repository)
        query = query.strip()
        if not query:
            return "Error: query must not be empty."
        search_query = f"{query} repo:{GITHUB_ORGANIZATION}/{repository}"
        if path:
            search_query += f" path:{_validate_path(path)}"
        result = await _github_get(
            "/search/code", {"q": search_query, "per_page": "10"}
        )
        matches = [
            {
                "path": item.get("path"),
                "url": item.get("html_url"),
                "repository": repository,
            }
            for item in result.get("items", [])
        ]
        return json.dumps({"repository": repository, "results": matches}, indent=2)
    except ValueError as exc:
        return f"Error: {exc}"
    except httpx.HTTPError as exc:
        return f"Error searching GitHub source: {exc}"


@tool
async def read_langchain_source(repository: str, path: str, ref: str = "") -> str:
    """Read a file from an allowlisted public langchain-ai repository."""
    try:
        repository = _validate_repository(repository)
        path = _validate_path(path)
        endpoint = f"/repos/{GITHUB_ORGANIZATION}/{repository}/contents/{path}"
        params = {"ref": ref} if ref.strip() else None
        result = await _github_get(endpoint, params)
        if result.get("type") != "file":
            return "Error: path does not identify a single file."
        content = result.get("content", "")
        decoded_content = base64.b64decode(content).decode("utf-8") if content else ""
        return json.dumps(
            {
                "repository": repository,
                "path": path,
                "ref": result.get("sha") if not ref.strip() else ref,
                "url": result.get("html_url"),
                "content": decoded_content,
            },
            indent=2,
        )
    except ValueError as exc:
        return f"Error: {exc}"
    except UnicodeDecodeError as exc:
        return f"Error decoding GitHub source: {exc}"
    except httpx.HTTPError as exc:
        return f"Error reading GitHub source: {exc}"
