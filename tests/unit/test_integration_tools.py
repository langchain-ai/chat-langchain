"""Tests for the integration page finder."""

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
        ("ChatAnthropic", "python", "/oss/python/integrations/chat/anthropic"),
        (
            "TavilySearchResults",
            "python",
            "/oss/python/integrations/tools/tavily_search",
        ),
        ("TavilySearch", "python", "/oss/python/integrations/tools/tavily_search"),
        ("TavilyExtract", "python", "/oss/python/integrations/tools/tavily_extract"),
        ("TavilyCrawl", "python", "/oss/python/integrations/tools/tavily_crawl"),
        ("TavilyMap", "python", "/oss/python/integrations/tools/tavily_map"),
        ("ExaSearchResults", "python", "/oss/python/integrations/tools/exa_search"),
        ("ExaSearchRetriever", "python", "/oss/python/integrations/tools/exa_search"),
        ("Neo4jGraph", "python", "/oss/python/integrations/graphs/neo4j_cypher"),
        (
            "GraphCypherQAChain",
            "python",
            "/oss/python/integrations/graphs/neo4j_cypher",
        ),
        ("ChatBedrockConverse", "python", "/oss/python/integrations/chat/bedrock"),
        (
            "PineconeVectorStore",
            "python",
            "/oss/python/integrations/vectorstores/pinecone",
        ),
        ("tavily", "python", "/oss/python/integrations/providers/tavily"),
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
async def test_read_returns_page_and_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(path: str) -> str:
        return f"# page {path}"

    monkeypatch.setattr(integration_tools, "_fetch_page", fake_fetch)
    out = await read_integration_page.ainvoke(
        {"name": "DoclingLoader", "language": "python"}
    )
    assert out.startswith(
        "Page: https://docs.langchain.com/oss/python/integrations/document_loaders/docling"
    )
    assert "# page /oss/python/integrations/document_loaders/docling" in out


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


@pytest.mark.asyncio
async def test_no_match_suggests_only_positive_language_candidates(monkeypatch) -> None:
    entries = [
        {"path": f"/oss/python/integrations/tools/near_{i}", "classes": ["NearClass"]}
        for i in range(4)
    ] + [
        {"path": "/oss/javascript/integrations/tools/near", "classes": ["NearClass"]},
        {"path": "/oss/python/integrations/tools/unrelated", "classes": []},
    ]
    monkeypatch.setattr(integration_tools, "_load_index", lambda: entries)
    out = await read_integration_page.ainvoke({"name": "NearClass"})
    assert "No dedicated python integration page" in out
    assert out.split("Closest matching paths:\n")[1].splitlines() == [
        f"- /oss/python/integrations/tools/near_{i}" for i in range(3)
    ]


@pytest.mark.asyncio
async def test_no_match_without_positive_candidates() -> None:
    out = await read_integration_page.ainvoke({"name": "zzzzunknown"})
    assert out == (
        "No dedicated python integration page matches 'zzzzunknown'. "
        "Use the docs search results / integrations index pages instead."
    )


def test_verified_class_does_not_promote_provider_or_incidental_class(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        integration_tools,
        "_load_index",
        lambda: [
            {
                "path": "/oss/python/integrations/providers/unrelated",
                "verified_classes": ["TargetClass"],
            },
            {
                "path": "/oss/python/integrations/tools/unrelated",
                "classes": ["TargetClass"],
            },
            {
                "path": "/oss/python/integrations/tools/actual",
                "verified_classes": ["TargetClass"],
            },
        ],
    )
    assert _paths("TargetClass") == ["/oss/python/integrations/tools/actual"]


@pytest.mark.parametrize(
    ("path", "text", "expected"),
    [
        (
            "tools/tavily_search.mdx",
            """
import { TavilySearchComponent } from "@/components/tavily";
```python
from langchain_tavily import (
    TavilySearch,
    TavilyExtract,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
```
""",
            ["TavilyExtract", "TavilySearch"],
        ),
        (
            "graphs/neo4j_cypher.mdx",
            """
```python
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
```
""",
            ["GraphCypherQAChain", "Neo4jGraph"],
        ),
        (
            "tools/exa_search.mdx",
            """
```typescript
import {
  ExaSearchResults as Search,
  ExaRetriever,
} from "@langchain/exa";
import { ChatOpenAI } from "@langchain/openai";
import { ExaComponent } from "@/components/exa";
```
""",
            ["ExaRetriever", "ExaSearchResults"],
        ),
    ],
)
def test_extract_integration_imports(path, text, expected) -> None:
    assert _extract_classes(text, path, "", "") == expected


@pytest.mark.parametrize(
    ("slug", "query", "matches"),
    [
        ("custom_search", "CustomSearchResults", True),
        ("custom_search", "OtherCustomSearchResults", False),
        ("tiny", "TinyResults", False),
    ],
)
def test_slug_prefix_requires_five_characters(
    monkeypatch, slug, query, matches
) -> None:
    path = f"/oss/python/integrations/tools/{slug}"
    monkeypatch.setattr(integration_tools, "_load_index", lambda: [{"path": path}])
    assert _paths(query) == ([path] if matches else [])
