"""Read-only source lookup tools for public LangChain repositories."""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import quote, urlparse

import httpx
from langchain.tools import tool

RepoName = Literal["langgraph", "langchain", "deepagents"]

GITHUB_API_HOST = "api.github.com"
GITHUB_RAW_HOST = "raw.githubusercontent.com"
GITHUB_OWNER = "langchain-ai"
ALLOWED_REPOS = frozenset(("langgraph", "langchain", "deepagents"))
MAX_SOURCE_BYTES = 100_000
MAX_SEARCH_RESULTS = 10
REQUEST_TIMEOUT = 15.0


def _token() -> str | None:
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_repo(repo: str) -> None:
    if repo not in ALLOWED_REPOS:
        raise ValueError("Repository is not allowlisted")


def _validate_path(path: str) -> None:
    if not path or path.startswith(("/", "\\")) or "\\" in path or "\x00" in path:
        raise ValueError("Invalid source path")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError("Invalid source path")


def _validate_url(url: str, host: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != host or parsed.port is not None:
        raise ValueError("URL host is not allowlisted")
    allowed_prefix = "/langchain-ai/" if host == GITHUB_RAW_HOST else "/"
    if host == GITHUB_RAW_HOST and not parsed.path.startswith(allowed_prefix):
        raise ValueError("URL owner is not allowlisted")
    if host == GITHUB_API_HOST and parsed.path.startswith("/repos/") and not parsed.path.startswith(
        f"/repos/{GITHUB_OWNER}/"
    ):
        raise ValueError("URL owner is not allowlisted")


def _error(response: httpx.Response) -> str:
    if response.status_code in (403, 429):
        return "GitHub rate limit reached; try again later."
    return f"GitHub request failed with status {response.status_code}."


def _request(client: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
    host = urlparse(url).hostname
    if host not in (GITHUB_API_HOST, GITHUB_RAW_HOST):
        raise ValueError("URL host is not allowlisted")
    _validate_url(url, host)
    return client.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT, **kwargs)


def _search_source(query: str, repo: RepoName) -> str:
    _validate_repo(repo)
    if not query.strip():
        return "Error: Search query cannot be empty."
    url = f"https://{GITHUB_API_HOST}/search/code"
    params = {
        "q": f"{query.strip()} org:{GITHUB_OWNER} repo:{GITHUB_OWNER}/{repo}",
        "per_page": str(MAX_SEARCH_RESULTS),
    }
    try:
        with httpx.Client(follow_redirects=False) as client:
            response = _request(client, url, params=params)
            if response.status_code >= 400:
                return _error(response)
            payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError):
        return "GitHub source search failed; try again later."
    if not isinstance(payload, dict):
        return "GitHub source search returned an invalid response."

    results = []
    for item in payload.get("items", []):
        path = item.get("path")
        html_url = item.get("html_url")
        if not isinstance(path, str) or not isinstance(html_url, str):
            continue
        snippet = ""
        matches = item.get("text_matches", [])
        if matches and isinstance(matches[0], dict):
            snippet = str(matches[0].get("fragment", ""))[:300]
        results.append(f"- {path}\n  {html_url}\n  {snippet}".rstrip())
    return "\n".join(results) if results else "No matching source files found."


@tool
def search_langchain_source(query: str, repo: RepoName) -> str:
    """Search allowlisted public LangChain source repositories."""
    try:
        return _search_source(query, repo)
    except ValueError as exc:
        return f"Error: {exc}"


def _read_source(repo: RepoName, path: str, start_line: int, end_line: int) -> str:
    _validate_repo(repo)
    _validate_path(path)
    if start_line < 1 or end_line < start_line:
        return "Error: Invalid line range."

    metadata_url = f"https://{GITHUB_API_HOST}/repos/{GITHUB_OWNER}/{repo}"
    try:
        with httpx.Client(follow_redirects=False) as client:
            metadata_response = _request(client, metadata_url)
            if metadata_response.status_code >= 400:
                return _error(metadata_response)
            default_branch = metadata_response.json().get("default_branch")
            if not isinstance(default_branch, str) or not default_branch:
                return "GitHub repository metadata did not include a default branch."
            encoded_branch = quote(default_branch, safe="")
            encoded_path = quote(path, safe="/")
            raw_url = (
                f"https://{GITHUB_RAW_HOST}/{GITHUB_OWNER}/{repo}/"
                f"{encoded_branch}/{encoded_path}"
            )
            _validate_url(raw_url, GITHUB_RAW_HOST)
            response = _request(client, raw_url)
            if response.status_code >= 400:
                return _error(response)
            content = response.content[:MAX_SOURCE_BYTES].decode("utf-8", errors="replace")
    except (httpx.HTTPError, ValueError, TypeError):
        return "GitHub source file read failed; try again later."

    lines = content.splitlines()
    selected = lines[start_line - 1 : end_line]
    suffix = "\n[output truncated at byte limit]" if len(response.content) > MAX_SOURCE_BYTES else ""
    numbered = "\n".join(
        f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start_line)
    )
    github_url = (
        f"https://github.com/{GITHUB_OWNER}/{repo}/blob/"
        f"{quote(default_branch, safe='')}/{quote(path, safe='/')}"
    )
    return f"{github_url}#L{start_line}-L{min(end_line, len(lines))}\n{numbered}{suffix}"


@tool
def read_langchain_source(
    repo: RepoName,
    path: str,
    start_line: int = 1,
    end_line: int = 120,
) -> str:
    """Read a bounded line range from an allowlisted public LangChain repository."""
    try:
        return _read_source(repo, path, start_line, end_line)
    except ValueError as exc:
        return f"Error: {exc}"
