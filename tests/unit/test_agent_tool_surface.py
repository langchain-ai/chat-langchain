import os
from unittest.mock import patch

from langchain_core.tools import StructuredTool
from managed_deepagents.runtime import compile_managed_agent

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


def _tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(
        lambda: None,
        name=name,
        description=f"Test tool {name}",
    )


def test_compiled_agent_exposes_only_documentation_tools():
    from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES

    from agent import agent, docs_agent_tools

    mcp_tools = [_tool("search_docs"), _tool("fetch_docs")]
    with patch("managed_deepagents.runtime.load_connector_tools", return_value=mcp_tools):
        compiled = compile_managed_agent(
            agent,
            connectors=[object()],
        )

    graph = compiled.get_graph()
    tool_names = set(graph.nodes["tools"].data._tools_by_name)
    profile = _HARNESS_PROFILES["google_genai:gemini-3.5-flash-lite"]
    tool_names -= profile.excluded_tools
    tool_names.discard("task")

    assert tool_names == {
        *(tool.name for tool in docs_agent_tools),
        "search_docs",
        "fetch_docs",
    }

    assert tool_names.isdisjoint(profile.excluded_tools | {"task"})
