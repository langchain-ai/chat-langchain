"""Tests for bounded documentation search results."""

import asyncio
import json

from langchain_core.tools import StructuredTool

from connectors.mcp import DOCS_SEARCH_TOOL_NAME, _bounded_tool


def test_bounded_docs_search_tool_keeps_ranked_metadata_under_budget():
    result = {
        "results": [
            {
                "title": f"Result {index}",
                "link": f"https://docs.langchain.com/result-{index}",
                "page": f"docs/result-{index}.mdx",
                "content": "x" * 5000,
            }
            for index in range(12)
        ]
    }

    async def search(query: str) -> dict:
        return result

    tool = StructuredTool.from_function(
        coroutine=search,
        name=DOCS_SEARCH_TOOL_NAME,
        description="Search docs",
    )
    bounded_tool = _bounded_tool(tool, max_chars=8000)
    serialized = asyncio.run(bounded_tool.ainvoke({"query": "agents"}))
    bounded = json.loads(serialized)

    assert len(serialized) <= 8000
    assert len(bounded["results"]) == 8
    assert bounded["results"][0]["Title"] == "Result 0"
    assert bounded["results"][0]["Link"] == "https://docs.langchain.com/result-0"
    assert bounded["results"][0]["Page"] == "docs/result-0.mdx"
    assert bounded["note"].startswith("Full page content")
