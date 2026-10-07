import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_coding_agents_use_docs_mcp_not_workspace_data_servers():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "## Connecting these docs to coding agents" in instructions
    assert (
        "Read `/use-these-docs.mdx` with `query_docs_filesystem_docs_by_lang_chain`"
        in instructions
    )
    assert "https://docs.langchain.com/mcp" in instructions
    assert "https://reference.langchain.com/mcp" in instructions
    assert (
        "claude mcp add --transport http docs-langchain https://docs.langchain.com/mcp"
        in instructions
    )
    assert "https://docs.langchain.com/use-these-docs" in instructions
    assert (
        "LangSmith Remote MCP (`/langsmith/langsmith-remote-mcp`) exposes workspace data such as traces and datasets"
        in instructions
    )
    assert (
        "Use it for workspace-data questions, not docs-access requests" in instructions
    )
    assert (
        "Do not recommend generic MCP test servers as a docs-access solution"
        in instructions
    )


def test_runtime_passes_shared_instructions_to_managed_agent(monkeypatch):
    root = Path(__file__).parents[2]
    instructions = (root / "instructions.md").read_text()
    compile_agent = Mock()
    definition = object()
    connectors = object()
    identity = object()
    modules = {
        "managed_deepagents.runtime": SimpleNamespace(
            compile_managed_agent=compile_agent
        ),
        "agent": SimpleNamespace(agent=definition),
        "_mda_connectors": SimpleNamespace(connectors=connectors),
        "identity": SimpleNamespace(identity=identity),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    runtime = runpy.run_path(str(root / "_mda_entry.py"))
    config: dict[str, object] = {}
    runtime["agent"](config)

    compile_agent.assert_called_once_with(
        definition,
        config,
        system_prompt=instructions,
        connectors=connectors,
        identity=identity,
    )
