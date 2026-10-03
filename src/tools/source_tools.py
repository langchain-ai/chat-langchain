"""Read-only tools for inspecting supported LangChain GitHub repositories."""

import base64
import os
from typing import Any

import requests
from langchain.tools import tool

GITHUB_API_URL = "https://api.github.com"
GITHUB_REPOSITORIES = frozenset(
    {
        "langchain-ai/langchain",
        "langchain-ai/langgraph",
        "langchain-ai/deepagents",
        "langchain-ai/langsmith-sdk",
    }
)
MAX_SEARCH_RESULTS = 10
MAX_FILE_CONTENT_CHARS = 12_000
REQUEST_TIMEOUT = 10.0


def _validate_repository(repo: str) -> str | None:
    if repo not in GITHUB_REPOSITORIES:
        allowed = ", ".join(sorted(GITHUB_REPOSITORIES))
        return f"Error: repository must be one of: {allowed}."
    return None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_error(response: requests.Response) -> str | None:
    if response.status_code in {403, 429}:
        return "Error: GitHub rate limit reached. Try again later."
    if response.status_code == 404:
        return "Error: GitHub repository or path was not found."
    if response.status_code >= 400:
        return f"Error: GitHub request failed with HTTP {response.status_code}."
    return None


def _format_source_url(repo: str, path: str, ref: str | None = None) -> str:
    branch = ref or "main"
    return f"https://github.com/{repo}/blob/{branch}/{path}"


@tool
def search_langchain_source(query: str, repo: str) -> str:
    """Search source files in an allowed LangChain GitHub repository."""
    validation_error = _validate_repository(repo)
    if validation_error:
        return validation_error
    if not query.strip():
        return "Error: query must not be empty."

    try:
        response = requests.get(
            f"{GITHUB_API_URL}/search/code",
            headers=_github_headers(),
            params={
                "q": f"{query.strip()} repo:{repo}",
                "per_page": str(MAX_SEARCH_RESULTS),
            },
            timeout=REQUEST_TIMEOUT,
        )
        request_error = _request_error(response)
        if request_error:
            return request_error
        payload: dict[str, Any] = response.json()
    except requests.RequestException:
        return "Error: GitHub request failed. Try again later."
    except ValueError:
        return "Error: GitHub returned an invalid response."

    items = payload.get("items", [])[:MAX_SEARCH_RESULTS]
    if not items:
        return f"No source files found for {query!r} in {repo}."

    lines = [f"Source search results for {query!r} in {repo}:"]
    for item in items:
        path = item.get("path", "")
        url = item.get("html_url") or _format_source_url(repo, path)
        lines.append(f"- {path}: {url}")
    return "\n".join(lines)


@tool
def read_langchain_source(repo: str, path: str) -> str:
    """Read a bounded source file from an allowed LangChain GitHub repository."""
    validation_error = _validate_repository(repo)
    if validation_error:
        return validation_error
    if not path.strip():
        return "Error: path must not be empty."

    try:
        response = requests.get(
            f"{GITHUB_API_URL}/repos/{repo}/contents/{path.lstrip('/')}",
            headers=_github_headers(),
            params={"ref": "main"},
            timeout=REQUEST_TIMEOUT,
        )
        request_error = _request_error(response)
        if request_error:
            return request_error
        payload: dict[str, Any] = response.json()
    except requests.RequestException:
        return "Error: GitHub request failed. Try again later."
    except ValueError:
        return "Error: GitHub returned an invalid response."

    encoded_content = payload.get("content")
    if not isinstance(encoded_content, str):
        return "Error: GitHub did not return readable file content."
    try:
        content = base64.b64decode(encoded_content, validate=False).decode(
            "utf-8", errors="replace"
        )
    except (ValueError, UnicodeError):
        return "Error: GitHub returned unreadable file content."

    truncated = len(content) > MAX_FILE_CONTENT_CHARS
    if truncated:
        content = content[:MAX_FILE_CONTENT_CHARS]
    url = payload.get("html_url") or _format_source_url(repo, path.lstrip("/"))
    suffix = f"\n[Content truncated to {MAX_FILE_CONTENT_CHARS} characters.]" if truncated else ""
    return f"Source file: {url}\n\n{content}{suffix}"
