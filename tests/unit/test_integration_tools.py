"""Tests for the integration page finder."""

import json
from pathlib import Path

import httpx
import pytest

from scripts import build_integration_index
from scripts.build_integration_index import _extract_classes
from src.tools import integration_tools
from src.tools.integration_tools import find_candidates, read_integration_page


def _paths(name: str, language: str = "python") -> list[str]:
    return [c["path"] for c in find_candidates(name, language)]


@pytest.mark.parametrize(
    ("name", "language", "expected"),
    [
        (
            "DoclingLoader",
            "python",
            "/oss/python/integrations/document_loaders/docling",
        ),
        (
            "langchain-docling",
            "python",
            "/oss/python/integrations/document_loaders/docling",
        ),
        ("ChatOllama", "javascript", "/oss/javascript/integrations/chat/ollama"),
        ("ollama", "javascript", "/oss/javascript/integrations/chat/ollama"),
        ("ChatAnthropic", "python", "/oss/python/integrations/chat/anthropic"),
        ("Neo4jGraph", "python", "/oss/python/integrations/graphs/neo4j_cypher"),
        (
            "TavilySearchResults",
            "python",
            "/oss/python/integrations/tools/tavily_search",
        ),
        (
            "langchain-google-genai",
            "python",
            "/oss/python/integrations/chat/google_generative_ai",
        ),
        ("PGVector", "python", "/oss/python/integrations/vectorstores/pgvector"),
        (
            "https://docs.langchain.com/oss/javascript/integrations/document_loaders/file_loaders/multi_file",
            "javascript",
            "/oss/javascript/integrations/document_loaders/file_loaders/multi_file",
        ),
        (
            "multi_file",
            "javascript",
            "/oss/javascript/integrations/document_loaders/file_loaders/multi_file",
        ),
        (
            "MultiFileLoader",
            "javascript",
            "/oss/javascript/integrations/document_loaders/file_loaders/multi_file",
        ),
        (
            "DirectoryLoader",
            "javascript",
            "/oss/javascript/integrations/document_loaders/file_loaders/directory",
        ),
    ],
)
def test_top_match(name: str, language: str, expected: str) -> None:
    assert _paths(name, language)[0] == expected


def test_language_filter() -> None:
    assert all(p.startswith("/oss/python/") for p in _paths("ChatOllama", "python"))


def test_nested_language_filter() -> None:
    url = "https://docs.langchain.com/oss/javascript/integrations/document_loaders/file_loaders/multi_file"
    assert _paths(url, "javascript") == [url.removeprefix("https://docs.langchain.com")]
    assert _paths(url, "python") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("slug", ["multi_file", "directory"])
async def test_read_nested_page(monkeypatch: pytest.MonkeyPatch, slug: str) -> None:
    path = f"/oss/javascript/integrations/document_loaders/file_loaders/{slug}"

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"https://docs.langchain.com{path}.md"
        return httpx.Response(
            200, text="> ## Documentation Index\n\n# Loader documentation"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        integration_tools.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(integration_tools, "_page_cache", {})
    out = await read_integration_page.ainvoke(
        {"name": f"https://docs.langchain.com{path}", "language": "javascript"}
    )
    assert out.startswith(f"Page: https://docs.langchain.com{path}\n")
    assert "# Loader documentation" in out
    assert "Documentation Index" not in out


@pytest.mark.parametrize("name", ["PyPDFLoader", "WebBaseLoader", "", "x"])
def test_incidental_or_missing_names_do_not_match(name: str) -> None:
    assert _paths(name) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("DoclingLoader", "/oss/python/integrations/document_loaders/docling"),
        ("Neo4jGraph", "/oss/python/integrations/graphs/neo4j_cypher"),
        (
            "https://docs.langchain.com/oss/python/integrations/graphs/neo4j_cypher",
            "/oss/python/integrations/graphs/neo4j_cypher",
        ),
        ("TavilySearchResults", "/oss/python/integrations/tools/tavily_search"),
        (
            "langchain-google-genai",
            "/oss/python/integrations/chat/google_generative_ai",
        ),
        ("ChatOllama", "/oss/python/integrations/chat/ollama"),
    ],
)
async def test_read_returns_page_and_alternatives(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    path: str,
) -> None:
    async def fake_fetch(path: str) -> str:
        return f"# page {path}"

    monkeypatch.setattr(integration_tools, "_fetch_page", fake_fetch)
    out = await read_integration_page.ainvoke({"name": name, "language": "python"})
    assert out.startswith(f"Page: https://docs.langchain.com{path}\n")
    assert f"# page {path}" in out


@pytest.mark.asyncio
async def test_read_fetch_failure_is_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(path: str) -> str:
        raise RuntimeError("down")

    monkeypatch.setattr(integration_tools, "_fetch_page", boom)
    out = await read_integration_page.ainvoke(
        {"name": "DoclingLoader", "language": "python"}
    )
    assert "could not be read" in out


@pytest.mark.asyncio
async def test_no_match_message() -> None:
    out = await read_integration_page.ainvoke(
        {"name": "PyPDFLoader", "language": "python"}
    )
    assert "No dedicated python integration page" in out


def test_strip_preamble() -> None:
    md = "> ## Documentation Index\n> Fetch ...\n\n# Docling integration\nbody"
    assert integration_tools._strip_preamble(md).startswith("# Docling integration")


@pytest.mark.parametrize(
    "reference",
    [
        "https://docs.langchain.com/oss/python/integrations/graphs/neo4j_cypher",
        "https://docs.langchain.com/oss/python/integrations/graphs/neo4j_cypher.mdx?x=1#usage",
        "/oss/python/integrations/graphs/neo4j_cypher.md",
        "oss/python/integrations/graphs/neo4j_cypher/",
        "neo4j_cypher",
    ],
)
def test_page_references(reference: str) -> None:
    assert _paths(reference)[0] == "/oss/python/integrations/graphs/neo4j_cypher"
    if "/" in reference:
        assert _paths(reference, "javascript") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [24, 25, 59, 60])
async def test_read_thresholds(monkeypatch: pytest.MonkeyPatch, score: int) -> None:
    entries = [
        {"path": f"/oss/python/integrations/tools/example{i}", "title": f"Example {i}"}
        for i in range(5)
    ]
    monkeypatch.setattr(integration_tools, "_load_index", lambda: entries)
    monkeypatch.setattr(integration_tools, "_score", lambda entry, query: score)

    async def fake_fetch(path: str) -> str:
        return "page body"

    monkeypatch.setattr(integration_tools, "_fetch_page", fake_fetch)
    out = await read_integration_page.ainvoke({"name": "example"})
    if score < 25:
        assert out.startswith("No dedicated python integration page")
    elif score < 60:
        assert out.startswith(
            "Closest integration pages: Example 0 (https://docs.langchain.com/"
        )
        assert out.count("https://docs.langchain.com/") == 3
        assert "page body" not in out
    else:
        assert out.startswith("Page: https://docs.langchain.com/")
        assert "page body" in out


@pytest.mark.asyncio
async def test_near_match_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_tools,
        "_load_index",
        lambda: [
            {
                "path": "/oss/python/integrations/tools/example_provider",
                "title": "Example provider",
            }
        ],
    )
    out = await read_integration_page.ainvoke({"name": "example_provider_extra"})
    url = "https://docs.langchain.com/oss/python/integrations/tools/example_provider"
    assert out == f"Closest integration pages: Example provider ({url})"
    assert _paths(url) == ["/oss/python/integrations/tools/example_provider"]


def test_generic_tokens_do_not_match_every_page() -> None:
    assert _paths("langchain-chat-graph-chain-search-results-loader-tool-toolkit") == []


def test_extract_integration_classes() -> None:
    text = """
    Neo4jGraph GraphQAChain TavilyTool TavilyToolkit MemorySaver InMemoryStore ModelEndpoint
    from langchain_provider import CustomLLM, CustomAPI
    from langchain_provider import (
        CustomModel,
        CustomEmbeddings,
    )
    """
    assert set(_extract_classes(text)) == {
        "Neo4jGraph",
        "GraphQAChain",
        "TavilyTool",
        "TavilyToolkit",
        "MemorySaver",
        "InMemoryStore",
        "ModelEndpoint",
        "CustomLLM",
        "CustomAPI",
        "CustomModel",
        "CustomEmbeddings",
    }


def test_build_nested_integration_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = [
        "oss/python/integrations/chat/ollama",
        "oss/javascript/integrations/document_loaders/file_loaders/multi_file",
        "oss/javascript/integrations/document_loaders/web_loaders/langsmith",
        "oss/python/integrations/document_loaders/nested/example",
    ]
    excluded = [
        "oss/javascript/integrations/document_loaders/index",
        "oss/javascript/integrations/document_loaders/file_loaders/index",
        "oss/javascript/integrations/document_loaders/web_loaders/index",
        "oss/python/integrations/chat/index",
        "oss/javascript/integrations/index",
    ]
    output = tmp_path / "integration_pages.json"

    def fake_get(url: str) -> str:
        if "git/trees/" in url:
            return json.dumps(
                {"tree": [{"path": f"src/{p}.mdx"} for p in pages + excluded]}
            )
        assert any(url.endswith(f"src/{p}.mdx") for p in pages)
        return '---\ntitle: "Multiple files"\nintegration:\n  name: "MultiFileLoader"\n---\nMultiFileLoader'

    monkeypatch.setattr(build_integration_index, "_get", fake_get)
    monkeypatch.setattr(build_integration_index, "OUT", output)
    build_integration_index.main()
    entries = json.loads(output.read_text())
    assert {e["path"] for e in entries} == {f"/{p}" for p in pages}
    assert all(e["title"] == "Multiple files" for e in entries)
    assert all(e["name"] == "MultiFileLoader" for e in entries)
    assert all(e["classes"] == ["MultiFileLoader"] for e in entries)
