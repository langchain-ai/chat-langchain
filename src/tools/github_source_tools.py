"""Read-only GitHub source lookup tools for the public LangChain repositories."""

import base64
import json
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
from langchain.tools import tool

GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_ORG = "langchain-ai"
GITHUB_SOURCE_REPOSITORIES = frozenset(
    {"langchain", "langgraph", "deepagents", "langsmith-sdk"}
)
GITHUB_TIMEOUT = 15.0


def _validate_repository(repo: str) -> str:
    """Validate and return an allowlisted repository name."""
    repository = repo.strip()
    if repository not in GITHUB_SOURCE_REPOSITORIES:
        allowed = ", ".join(sorted(GITHUB_SOURCE_REPOSITORIES))
        raise ValueError(f"Repository must be one of: {allowed}")
    return repository


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers, including an optional token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _github_get(path: str, params: Mapping[str, str] | None = None) -> Any:
    """Fetch a JSON response from GitHub."""
    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE_URL,
        headers=_github_headers(),
        timeout=GITHUB_TIMEOUT,
    ) as client:
        response = await client.get(path, params=params)
        response.raise_for_status()
        return response.json()


async def _resolve_ref(repo: str, ref: str | None) -> str:
    """Resolve an omitted ref to the repository default branch."""
    if ref:
        return ref
    repository = await _github_get(f"/repos/{GITHUB_ORG}/{repo}")
    default_branch = repository.get("default_branch")
    if not default_branch:
        raise ValueError("GitHub did not return a default branch")
    return default_branch


def _repository_url(repo: str, path: str, ref: str) -> str:
    """Build a browser URL for a repository file."""
    return (
        f"https://github.com/{GITHUB_ORG}/{repo}/blob/"
        f"{quote(ref, safe='')}/{quote(path, safe='/')}"
    )


async def _search_github_source(query: str, repo: str | None = None) -> str:
    """Search allowlisted GitHub repositories for source code."""
    if not query.strip():
        return "No GitHub source matches: the search query is empty."

    repositories = (
        [_validate_repository(repo)]
        if repo is not None
        else sorted(GITHUB_SOURCE_REPOSITORIES)
    )
    matches: list[dict[str, str]] = []
    failures: list[str] = []
    for repository in repositories:
        try:
            result = await _github_get(
                "/search/code",
                params={
                    "q": f"{query.strip()} repo:{GITHUB_ORG}/{repository}",
                    "per_page": "10",
                },
            )
        except Exception as exc:
            failures.append(f"{repository}: {exc}")
            continue
        for item in result.get("items", []):
            path = item.get("path")
            html_url = item.get("html_url")
            if path and html_url:
                matches.append(
                    {
                        "repo": repository,
                        "path": path,
                        "url": html_url,
                    }
                )

    if matches:
        response: dict[str, Any] = {"matches": matches}
        if failures:
            response["unavailable"] = failures
        return json.dumps(response, indent=2)
    if failures:
        return "GitHub source search unavailable: " + "; ".join(failures)
    return "No GitHub source matches were found."


@tool
async def search_github_source(query: str, repo: str | None = None) -> str:
    """Search public LangChain source repositories and return matching paths and URLs."""
    try:
        return await _search_github_source(query, repo)
    except ValueError as exc:
        return f"GitHub source search error: {exc}"


async def _read_github_source(repo: str, path: str, ref: str | None = None) -> str:
    """Read a source file from an allowlisted GitHub repository."""
    repository = _validate_repository(repo)
    normalized_path = path.strip().strip("/")
    if not normalized_path:
        return "GitHub source read error: path must not be empty."
    resolved_ref = await _resolve_ref(repository, ref)
    encoded_path = quote(normalized_path, safe="/")
    result = await _github_get(
        f"/repos/{GITHUB_ORG}/{repository}/contents/{encoded_path}",
        params={"ref": resolved_ref},
    )
    if result.get("type") != "file":
        return f"GitHub source read error: {normalized_path} is not a file."
    content = result.get("content")
    if not isinstance(content, str):
        return "GitHub source read error: GitHub returned no file content."
    try:
        source = base64.b64decode(content).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        return f"GitHub source read error: could not decode file content ({exc})."
    url = result.get("html_url") or _repository_url(repository, normalized_path, resolved_ref)
    return json.dumps(
        {
            "repo": repository,
            "path": normalized_path,
            "ref": resolved_ref,
            "url": url,
            "content": source,
        },
        indent=2,
    )


@tool
async def read_github_source(repo: str, path: str, ref: str | None = None) -> str:
    """Read a source file from a public LangChain repository at an optional ref."""
    try:
        return await _read_github_source(repo, path, ref)
    except (ValueError, httpx.HTTPError) as exc:
        return f"GitHub source read unavailable: {exc}"
    except Exception as exc:
        return f"GitHub source read unavailable: {exc}"
