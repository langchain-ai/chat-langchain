"""Tests for the docs agent tool surface."""

import os
from dataclasses import replace

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from deepagents import graph
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from managed_deepagents.runtime import compile_managed_agent
from pydantic import PrivateAttr

import agent as docs_agent

FORBIDDEN_TOOLS = {
    "execute",
    "write_file",
    "edit_file",
    "delete",
    "ls",
    "read_file",
    "glob",
    "grep",
    "task",
}


class CapturingChatModel(FakeListChatModel):
    _captured_tool_names: list[str] = PrivateAttr()

    def __init__(self, *, captured_tool_names: list[str]) -> None:
        super().__init__(responses=["Done."])
        self._captured_tool_names = captured_tool_names

    def bind_tools(self, tools, **kwargs):
        self._captured_tool_names.extend(tool.name for tool in tools)
        return RunnableLambda(lambda _input: AIMessage(content="Done."))


def test_docs_agent_compiles_without_unused_builtin_tools(monkeypatch):
    """The compiled docs agent only binds authored and docs connector tools."""
    captured_tool_names: list[str] = []
    model = CapturingChatModel(captured_tool_names=captured_tool_names)

    @tool
    def search_docs_by_lang_chain(query: str) -> str:
        """Search LangChain documentation."""
        return query

    @tool
    def query_docs_filesystem_docs_by_lang_chain(path: str) -> str:
        """Read a LangChain documentation page."""
        return path

    monkeypatch.setattr(graph, "resolve_model", lambda _model: model)
    monkeypatch.setattr(
        "managed_deepagents.runtime.load_connector_tools",
        lambda _connectors: [
            search_docs_by_lang_chain,
            query_docs_filesystem_docs_by_lang_chain,
        ],
    )

    definition = replace(
        docs_agent.agent,
        config={**docs_agent.agent.config, "middleware": []},
    )
    compiled = compile_managed_agent(definition, connectors=[object()])
    compiled.invoke({"messages": [HumanMessage(content="Find the docs.")]})

    assert not FORBIDDEN_TOOLS.intersection(captured_tool_names)
    assert {
        "search_support_articles",
        "get_support_article_content",
        "fetch_langchain_pricing",
        "check_links",
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
    } <= set(captured_tool_names)
