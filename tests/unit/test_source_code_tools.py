"""Tests for the public LangChain source lookup tool."""

import asyncio
import base64
import json

import httpx

from src.tools.source_code_tools import search_langchain_source


class _MockAsyncClient:
    def __init__(self, responses):
        self.responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url):
        response = self.responses[url]
        response.request = httpx.Request("GET", url)
        return response


def _github_response(url, payload):
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def test_source_tool_rejects_unallowlisted_repository():
    result = asyncio.run(
        search_langchain_source.ainvoke({"repository": "other", "query": "streaming"})
    )

    assert "one of: langchain, langgraph, deepagents" in result


def test_source_tool_reads_allowlisted_file(monkeypatch):
    repository = "langchain-ai/langgraph"
    branch = "main"
    path = "libs/langgraph/langgraph/graph/state.py"
    content = "class StateGraph:\n    pass\n"
    encoded = base64.b64encode(content.encode()).decode()
    urls = {
        "https://api.github.com/repos/langchain-ai/langgraph": _github_response(
            "https://api.github.com/repos/langchain-ai/langgraph",
            {"default_branch": branch},
        ),
        f"https://api.github.com/repos/{repository}/contents/{path}?ref={branch}": _github_response(
            "https://example.test/file",
            {"encoding": "base64", "content": encoded, "size": len(content)},
        ),
    }
    monkeypatch.setattr(
        "src.tools.source_code_tools.httpx.AsyncClient",
        lambda **kwargs: _MockAsyncClient(urls),
    )

    result = json.loads(
        asyncio.run(
            search_langchain_source.ainvoke(
                {"repository": "langgraph", "operation": "read", "path": path}
            )
        )
    )

    assert result["repository"] == repository
    assert result["results"][0]["content"] == content
    assert result["results"][0]["url"].endswith(path)


def test_source_tool_limits_search_results(monkeypatch):
    repository = "langchain-ai/langchain"
    branch = "main"
    tree_url = (
        "https://api.github.com/repos/langchain-ai/langchain/git/trees/main?recursive=1"
    )
    tree = {
        "tree": [
            {"type": "blob", "path": f"libs/core/file-{index}.py", "size": 10}
            for index in range(8)
        ]
    }
    responses = {
        f"https://api.github.com/repos/{repository}": _github_response(
            f"https://api.github.com/repos/{repository}", {"default_branch": branch}
        ),
        tree_url: _github_response(tree_url, tree),
    }
    for index in range(5):
        path = f"libs/core/file-{index}.py"
        url = f"https://api.github.com/repos/{repository}/contents/{path}?ref={branch}"
        encoded = base64.b64encode(f"streaming = {index}\n".encode()).decode()
        responses[url] = _github_response(
            url, {"encoding": "base64", "content": encoded, "size": 15}
        )
    monkeypatch.setattr(
        "src.tools.source_code_tools.httpx.AsyncClient",
        lambda **kwargs: _MockAsyncClient(responses),
    )

    result = json.loads(
        asyncio.run(
            search_langchain_source.ainvoke(
                {"repository": "langchain", "query": "streaming", "max_results": 99}
            )
        )
    )

    assert len(result["results"]) == 5
