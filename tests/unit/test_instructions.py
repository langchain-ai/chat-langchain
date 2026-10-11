import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions

    assert (
        "Lead with a direct, bolded verdict only when it does; otherwise, never "
        "assert or deny the premise and lead with a bold sentence that says "
        'plainly "the documentation does not state X".'
    ) in instructions
    assert (
        "When neither verdict is supported by retrieved content from this "
        "thread, state that the documentation does not state the claim"
    ) in instructions


def test_bold_opening_is_conditional_on_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "**First sentence is bold and answers the question when the retrieved "
        "evidence supports it; otherwise, state that the documentation does "
        "not state the claim**"
    ) in instructions


def test_managed_runtime_receives_local_instructions(monkeypatch):
    root = Path(__file__).parents[2]
    compile_agent = Mock()
    definition = object()
    connectors = object()
    identity = object()
    modules = {
        "managed_deepagents.runtime": {"compile_managed_agent": compile_agent},
        "agent": {"agent": definition},
        "_mda_connectors": {"connectors": connectors},
        "identity": {"identity": identity},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    deployment = json.loads((root / "langgraph.json").read_text())
    assert deployment["graphs"]["docs_agent"] == "./_mda_entry.py:agent"

    entrypoint = runpy.run_path(str(root / "_mda_entry.py"))
    config = {"configurable": {}}

    assert entrypoint["agent"](config) is compile_agent.return_value
    compile_agent.assert_called_once_with(
        definition,
        config,
        system_prompt=(root / "instructions.md").read_text(),
        connectors=connectors,
        identity=identity,
    )
