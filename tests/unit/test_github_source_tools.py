import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.github_source_tools import (
    _github_headers,
    _read_github_source,
    _search_github_source,
)


def run(coroutine):
    return asyncio.run(coroutine)


def test_github_headers_use_optional_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    assert _github_headers()["Authorization"] == "Bearer test-token"


def test_search_rejects_unallowlisted_repository_before_network():
    github_get = AsyncMock()

    with patch("src.tools.github_source_tools._github_get", github_get):
        with pytest.raises(ValueError, match="Repository must be one of"):
            run(_search_github_source("middleware", "private-repo"))

    github_get.assert_not_awaited()


def test_search_returns_paths_and_urls():
    github_get = AsyncMock(
        return_value={
            "items": [
                {
                    "path": "libs/core/langchain/agents.py",
                    "html_url": "https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain/agents.py",
                }
            ]
        }
    )

    with patch("src.tools.github_source_tools._github_get", github_get):
        result = run(_search_github_source("create_agent", "langchain"))

    assert json.loads(result)["matches"] == [
        {
            "repo": "langchain",
            "path": "libs/core/langchain/agents.py",
            "url": "https://github.com/langchain-ai/langchain/blob/main/libs/core/langchain/agents.py",
        }
    ]
    github_get.assert_awaited_once_with(
        "/search/code",
        params={
            "q": "create_agent repo:langchain-ai/langchain",
            "per_page": "10",
        },
    )


def test_search_reports_api_failure_without_fabricating_matches():
    github_get = AsyncMock(side_effect=RuntimeError("rate limited"))

    with patch("src.tools.github_source_tools._github_get", github_get):
        result = run(_search_github_source("missing-symbol", "langchain"))

    assert "GitHub source search unavailable" in result
    assert "missing-symbol" not in result


def test_read_source_resolves_default_branch_and_decodes_content():
    source = "def create_agent():\n    return None\n"
    github_get = AsyncMock(
        side_effect=[
            {"default_branch": "trunk"},
            {
                "type": "file",
                "content": base64.b64encode(source.encode()).decode(),
                "html_url": "https://github.com/langchain-ai/langchain/blob/trunk/agent.py",
            },
        ]
    )

    with patch("src.tools.github_source_tools._github_get", github_get):
        result = run(_read_github_source("langchain", "agent.py"))

    assert json.loads(result) == {
        "repo": "langchain",
        "path": "agent.py",
        "ref": "trunk",
        "url": "https://github.com/langchain-ai/langchain/blob/trunk/agent.py",
        "content": source,
    }
    assert github_get.await_args_list[1].args == (
        "/repos/langchain-ai/langchain/contents/agent.py",
    )
    assert github_get.await_args_list[1].kwargs == {"params": {"ref": "trunk"}}


def test_read_source_honors_explicit_ref_without_resolving_default_branch():
    content = base64.b64encode(b"source").decode()
    github_get = AsyncMock(
        return_value={"type": "file", "content": content, "html_url": "https://example.test/file"}
    )

    with patch("src.tools.github_source_tools._github_get", github_get):
        result = run(_read_github_source("langgraph", "src/graph.py", "v1.0.0"))

    assert json.loads(result)["ref"] == "v1.0.0"
    github_get.assert_awaited_once_with(
        "/repos/langchain-ai/langgraph/contents/src/graph.py",
        params={"ref": "v1.0.0"},
    )


@pytest.mark.parametrize("repo", ["", "other-org/langchain", "langchain/../private"])
def test_read_source_rejects_invalid_repository_before_network(repo):
    github_get = AsyncMock()

    with patch("src.tools.github_source_tools._github_get", github_get):
        with pytest.raises(ValueError):
            run(_read_github_source(repo, "src/file.py"))

    github_get.assert_not_awaited()
