"""Tests for MCP connector wiring."""

from connectors.mcp import connector


def test_langchain_docs_server_declares_finite_timeout():
    """The LangChain docs MCP server must have a finite request timeout."""
    server_config = connector.config["mcp_servers"]["langchain-docs"]

    assert server_config["default_tool_timeout"] == 10
    assert 0 < server_config["default_tool_timeout"] < float("inf")
