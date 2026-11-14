"""Tests for the integration page finder."""

import pytest

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
