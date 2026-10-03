"""Read-only GitHub source tools for the supported LangChain repositories."""

import base64
import os
from typing import Literal
from urllib.parse import quote

import requests
from langchain.tools import tool
from pydantic import BaseModel

Repository = Literal["langchain", "langgraph", "deepagents"]

GITHUB_API_URL = "https://api.github.com"
GITHUB_ORG = "langchain-ai"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
REQUEST_TIMEOUT = 15
MAX_RESULTS = 10
MAX_SNIPPET_LENGTH = 300
MAX_READ_LINES = 200
MAX_OUTPUT_CHARS = 12_000
ALLOWED_REPOSITORIES = {"langchain", "langgraph", "deepagents"}


class _SearchSourceCodeInput(BaseModel):
    query: str
    repo: str


class _ReadSourceFileInput(BaseModel):
    repo: str
    path: str
    start_line: int = 1
    max_lines: int = MAX_READ_LINES


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "LangChain-Docs-Agent/1.0",
    }
    token = os.getenv(GITHUB_TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_repository(repo: str) -> str | None:
    if repo not in ALLOWED_REPOSITORIES:
        return "Invalid repository. Use langchain, langgraph, or deepagents."
    return None


def _validate_path(path: str) -> str | None:
    if not path or path.startswith(("/", "\\")) or "\\" in path:
        return "Invalid path. Use a repository-relative POSIX path."
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "Invalid path. Path traversal is not allowed."
    return None


def _request_error(response: requests.Response) -> str:
    if response.status_code in (403, 429):
        return "GitHub rate limit reached; source code is temporarily unavailable."
    if response.status_code == 404:
        return "GitHub resource not found."
    return f"GitHub request failed with HTTP {response.status_code}."


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS].rstrip() + "\n[Output truncated.]"


def _get(url: str, **kwargs: object) -> requests.Response | str:
    request_headers = _headers()
    request_headers.update(kwargs.pop("headers", {}))
    try:
        response = requests.get(url, headers=request_headers, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        return f"Unable to reach GitHub: {exc}"
    if not response.ok:
        return _request_error(response)
    return response


@tool(args_schema=_SearchSourceCodeInput)
def search_source_code(
    query: str, repo: Repository
) -> str:
    """Search implementation source in an allowlisted langchain-ai repository."""
    repository_error = _validate_repository(repo)
    if repository_error:
        return repository_error
    query = query.strip()
    if not query:
        return "Search query cannot be empty."
    if len(query) > 500:
        return "Search query is too long; limit it to 500 characters."

    response = _get(
        f"{GITHUB_API_URL}/search/code",
        params={"q": f"{query} repo:{GITHUB_ORG}/{repo}", "per_page": MAX_RESULTS},
        headers={"Accept": "application/vnd.github.text-match+json"},
    )
    if isinstance(response, str):
        return response
    try:
        items = response.json().get("items", [])[:MAX_RESULTS]
    except (TypeError, ValueError):
        return "GitHub returned an invalid search response."
    if not items:
        return "No source files matched that search."

    results = []
    for item in items:
        path = item.get("path", "unknown path")
        matches = item.get("text_matches", [])
        snippet = matches[0].get("fragment", "") if matches else "No snippet available."
        snippet = " ".join(snippet.split())[:MAX_SNIPPET_LENGTH]
        results.append(f"{path}\nSnippet: {snippet}\nURL: {item.get('html_url', '')}")
    return _truncate("\n\n".join(results))


@tool(args_schema=_ReadSourceFileInput)
def read_source_file(
    repo: Repository, path: str, start_line: int = 1, max_lines: int = 200
) -> str:
    """Read a bounded line range from an allowlisted langchain-ai repository."""
    repository_error = _validate_repository(repo)
    if repository_error:
        return repository_error
    path_error = _validate_path(path)
    if path_error:
        return path_error
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        return "start_line must be a positive integer."
    if isinstance(max_lines, bool) or not isinstance(max_lines, int) or not 1 <= max_lines <= MAX_READ_LINES:
        return f"max_lines must be an integer between 1 and {MAX_READ_LINES}."

    repository_url = f"{GITHUB_API_URL}/repos/{GITHUB_ORG}/{repo}"
    metadata = _get(repository_url)
    if isinstance(metadata, str):
        return metadata
    try:
        default_branch = metadata.json()["default_branch"]
    except (KeyError, TypeError, ValueError):
        return "GitHub returned an invalid repository response."

    content_url = f"{repository_url}/contents/{quote(path, safe='/')}"
    response = _get(content_url, params={"ref": default_branch})
    if isinstance(response, str):
        return response
    try:
        payload = response.json()
        if payload.get("type") != "file":
            return "GitHub path is not a file."
        content = base64.b64decode(payload["content"]).decode("utf-8")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        return "GitHub returned invalid file content."

    lines = content.splitlines()
    selected = lines[start_line - 1 : start_line - 1 + max_lines]
    if not selected:
        return "The requested line range is outside the file."
    numbered = "\n".join(
        f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start_line)
    )
    blob_path = quote(path, safe="/")
    blob_branch = quote(default_branch, safe="/")
    blob_url = f"https://github.com/{GITHUB_ORG}/{repo}/blob/{blob_branch}/{blob_path}"
    return _truncate(f"{numbered}\nGitHub URL: {blob_url}")
