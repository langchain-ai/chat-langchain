"""Tests for the integration page finder."""

import re

import pytest

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
        (
            "Perplexity search",
            "python",
            "/oss/python/integrations/tools/perplexity_search",
        ),
        (
            "perplexity_search",
            "python",
            "/oss/python/integrations/tools/perplexity_search",
        ),
        (
            "PerplexitySearchResults",
            "python",
            "/oss/python/integrations/tools/perplexity_search",
        ),
        (
            "Perplexity search tool",
            "python",
            "/oss/python/integrations/tools/perplexity_search",
        ),
        ("ChatOpenAI", "python", "/oss/python/integrations/chat/openai"),
        ("DeepSeek", "python", "/oss/python/integrations/chat/deepseek"),
        ("bedrock", "python", "/oss/python/integrations/chat/bedrock"),
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
    ],
)
def test_top_match(name: str, language: str, expected: str) -> None:
    assert _paths(name, language)[0] == expected


def test_language_filter() -> None:
    assert all(p.startswith("/oss/python/") for p in _paths("ChatOllama", "python"))


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


@pytest.mark.asyncio
async def test_stem_only_match_does_not_read_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_fetch(path: str) -> str:
        pytest.fail(f"A stem-only match must not read {path}")

    monkeypatch.setattr(integration_tools, "_fetch_page", unexpected_fetch)
    assert _paths("TextLoader") == []
    out = await read_integration_page.ainvoke({"name": "TextLoader"})
    assert out.startswith(("Closest integration pages:", "No dedicated python"))
    assert not out.startswith("Page:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "Perplexity search",
        "perplexity_search",
        "PerplexitySearchResults",
        "Perplexity search tool",
    ],
)
async def test_alternate_page_url_reads_exact_page(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    async def fake_fetch(path: str) -> str:
        return f"# page {path}"

    monkeypatch.setattr(integration_tools, "_fetch_page", fake_fetch)
    out = await read_integration_page.ainvoke({"name": name})
    assert out.startswith(
        "Page: https://docs.langchain.com/oss/python/integrations/tools/perplexity_search\n"
    )
    urls = re.findall(r"^- (https://docs\.langchain\.com/\S+)", out, re.MULTILINE)
    assert urls
    for url in urls:
        page = await read_integration_page.ainvoke({"name": url})
        assert page.startswith(f"Page: {url}\n")
        assert f"# page {url.removeprefix(integration_tools.DOCS_BASE_URL)}" in page
