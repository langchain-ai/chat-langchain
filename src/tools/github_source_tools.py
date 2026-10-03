"""Read-only tools for searching the public LangChain GitHub repositories."""

import json
import os
from typing import Literal

import httpx
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_REPOSITORIES = frozenset({"langchain-ai/langchain", "langchain-ai/langgraph"})
GITHUB_API_URL = "https://api.github.com"
GITHUB_RAW_URL = "https://raw.githubusercontent.com"
MAX_OUTPUT_CHARS = 5000
MAX_SEARCH_RESULTS = 10
MAX_LINE_LIMIT = 200
REQUEST_TIMEOUT = 15.0


class GitHubSourceInput(BaseModel):
    """Validate source search and read requests."""

    operation: Literal["search", "read"]
    query: str | None = Field(default=None, max_length=200)
    repo: str | None = None
    path: str | None = Field(default=None, max_length=300)
    line_limit: int = Field(default=100, ge=1, le=MAX_LINE_LIMIT)

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, value: str | None) -> str | None:
        """Validate the repository allowlist."""
        if value is not None and value not in ALLOWED_REPOSITORIES:
            raise ValueError(f"repo must be one of: {', '.join(sorted(ALLOWED_REPOSITORIES))}")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        """Validate a relative repository path."""
        if value is not None and (
            not value or value.startswith("/") or "\\" in value or ".." in value.split("/")
        ):
            raise ValueError("path must be a relative repository path without '..' segments")
        return value

    @model_validator(mode="after")
    def validate_operation_arguments(self) -> "GitHubSourceInput":
        """Validate arguments required by the selected operation."""
        if self.operation == "search" and not self.query:
            raise ValueError("query is required for search operations")
        if self.operation == "read" and (not self.repo or not self.path):
            raise ValueError("repo and path are required for read operations")
        return self


def _cap_text(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Limit tool output to a bounded number of characters."""
    if len(text) <= limit:
        return text
    suffix = "\n...[output truncated]"
    return f"{text[: limit - len(suffix)]}{suffix}"


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "LangChain-Docs-Agent/1.0",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _search_repository(client: httpx.AsyncClient, query: str, repo: str) -> list[dict[str, str]]:
    response = await client.get(
        f"{GITHUB_API_URL}/search/code",
        params={"q": f"{query} repo:{repo}", "per_page": MAX_SEARCH_RESULTS},
    )
    response.raise_for_status()
    return [
        {
            "repo": repo,
            "path": item["path"],
            "url": item.get("html_url", f"https://github.com/{repo}/blob/HEAD/{item['path']}"),
        }
        for item in response.json().get("items", [])
    ]


async def _search_source(query: str, repo: str | None) -> str:
    repositories = [repo] if repo else sorted(ALLOWED_REPOSITORIES)
    async with httpx.AsyncClient(headers=_headers(), timeout=REQUEST_TIMEOUT) as client:
        matches: list[dict[str, str]] = []
        for repository in repositories:
            matches.extend(await _search_repository(client, query, repository))
    if not matches:
        return "No source matches found."
    return _cap_text(json.dumps({"matches": matches}, indent=2))


async def _read_source(repo: str, path: str, line_limit: int) -> str:
    url = f"{GITHUB_RAW_URL}/{repo}/HEAD/{path}"
    async with httpx.AsyncClient(headers=_headers(), timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
    lines = response.text.splitlines()
    selected = lines[:line_limit]
    result = f"Repository: {repo}\nPath: {path}\nURL: https://github.com/{repo}/blob/HEAD/{path}\n\n"
    result += "\n".join(selected)
    if len(lines) > line_limit:
        result += f"\n...[{len(lines) - line_limit} more lines omitted]"
    return _cap_text(result)


@tool(args_schema=GitHubSourceInput)
async def search_langchain_source(
    operation: Literal["search", "read"],
    query: str | None = None,
    repo: str | None = None,
    path: str | None = None,
    line_limit: int = 100,
) -> str:
    """Search or read files from the public LangChain and LangGraph repositories."""
    try:
        if operation == "search":
            return await _search_source(query or "", repo)
        return await _read_source(repo or "", path or "", line_limit)
    except httpx.HTTPError as exc:
        return _cap_text(f"Source lookup failed: {exc}")
