"""Tests for the docs agent's compiled tool surface."""

from __future__ import annotations

import os

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("USE_LOCAL_PROMPTS", "1")

from langchain_core.tools import StructuredTool
from managed_deepagents import runtime

import agent


def test_compiled_agent_binds_only_authored_and_mcp_docs_tools(monkeypatch):
    """The compiled agent must not expose the default filesystem tools."""

    def search_docs(query: str) -> str:
        """Search documentation."""
        return query

    def read_docs(path: str) -> str:
        """Read documentation."""
        return path

    mcp_tools = [
        StructuredTool.from_function(
            search_docs,
            name="search_docs_by_lang_chain",
        ),
        StructuredTool.from_function(
            read_docs,
            name="query_docs_filesystem_docs_by_lang_chain",
        ),
    ]
    monkeypatch.setattr(runtime, "load_connector_tools", lambda _: mcp_tools)

    compiled_agent = runtime.compile_managed_agent(
        agent.agent,
        system_prompt="test",
        connectors=[object()],
    )

    bound_tool_names = set(compiled_agent.nodes["tools"].bound._tools_by_name)
    expected_tool_names = {tool.name for tool in agent.docs_agent_tools} | {
        tool.name for tool in mcp_tools
    }

    assert bound_tool_names == expected_tool_names
