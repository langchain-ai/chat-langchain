"""Test tool-name repair and bounded invalid-call recovery through an agent graph."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import Field

from src.middleware.tool_name_repair_middleware import ToolNameRepairMiddleware

DOCS = "langchainDocs_query_docs_filesystem_docs_by_lang_chain"


class RecordingModel(FakeMessagesListChatModel):
    tool_bindings: list[list] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tool_bindings.append(list(tools))
        return self


def call(name, index=0, args=None):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args or {"query": "langgraph"}, "id": str(index)}
        ],
    )


def graph_for(responses, names=(DOCS,)):
    def read(query: str) -> str:
        return "LangGraph documentation"

    model = RecordingModel(responses=responses)
    graph = create_agent(
        model=model,
        tools=[
            StructuredTool.from_function(read, name=name, description="Read docs")
            for name in names
        ],
        middleware=[ToolNameRepairMiddleware()],
    )
    return graph, model


def invoke(graph, asynchronous, messages=None):
    inputs = {"messages": messages or [HumanMessage("Find LangGraph documentation")]}
    config = {"recursion_limit": 12}
    return (
        asyncio.run(graph.ainvoke(inputs, config))
        if asynchronous
        else graph.invoke(inputs, config)
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "name, names, expected",
    [
        ("langgraphDocs_query_docs_filesystem_docs_by_lang_chain", (DOCS,), DOCS),
        (
            "langgraphDocs_query_docs_filesystem_docs_by_lang_chain =garbled",
            (DOCS,),
            DOCS,
        ),
        (DOCS + "=query", (DOCS,), DOCS),
        ("other_query_docs_filesystem_docs_by_lang_chain", (DOCS,), DOCS),
        (DOCS, (DOCS,), DOCS),
        ("unknown", (DOCS,), "unknown"),
        ("other_read", ("first_read", "second_read"), "other_read"),
    ],
)
def test_names_resolve_only_to_unique_bound_tools(
    asynchronous, name, names, expected, monkeypatch
):
    original = call(name)
    graph, _ = graph_for([original, AIMessage(content="Answer")], names)
    run = SimpleNamespace(metadata={})
    monkeypatch.setattr(
        "src.middleware.tool_name_repair_middleware.ls.get_current_run_tree",
        lambda: run,
    )
    state = invoke(graph, asynchronous)
    emitted = next(m for m in state["messages"] if isinstance(m, AIMessage))
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert emitted.tool_calls[0] == {**original.tool_calls[0], "name": expected}
    assert original.tool_calls[0]["name"] == name
    assert ("is not a valid tool" in result.text) == (expected not in names)
    assert bool(run.metadata.get("tool_name_repaired")) == (expected != name)
    assert state["messages"][-1].content == "Answer"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ignores_disabled_tools", [False, True])
def test_third_identical_failure_finishes_without_tools(
    asynchronous, ignores_disabled_tools
):
    final = (
        call("unknown", 3)
        if ignores_disabled_tools
        else AIMessage(content="Documentation unavailable")
    )
    graph, model = graph_for([call("unknown", i) for i in range(3)] + [final])
    state = invoke(graph, asynchronous)
    results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(results) == 2
    assert all("is not a valid tool" in m.text for m in results)
    assert not state["messages"][-1].tool_calls
    assert state["messages"][-1].text
    if not ignores_disabled_tools:
        assert state["messages"][-1].content == "Documentation unavailable"
    assert len(model.tool_bindings) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
def test_distinct_failures_and_previous_turn_do_not_trigger_guard(asynchronous):
    history = [HumanMessage("Earlier request")]
    for i in range(2):
        history.extend(
            [
                call("unknown", f"old-{i}"),
                ToolMessage(
                    content="unknown is not a valid tool", tool_call_id=f"old-{i}"
                ),
            ]
        )
    history.extend(
        [
            AIMessage(content="Earlier answer"),
            HumanMessage("Find LangGraph documentation"),
        ]
    )
    graph, model = graph_for(
        [
            call("unknown", 0),
            call("unknown", 1, {"query": "different"}),
            call("another", 2),
            call("unknown", 3),
            AIMessage(content="Answer"),
        ]
    )
    state = invoke(graph, asynchronous, history)
    assert state["messages"][-1].content == "Answer"
    assert len(model.tool_bindings) == 5
    assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 6


def test_repair_precedes_model_fallback():
    tree = ast.parse(Path("agent.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "docs_agent_middleware"
            for t in node.targets
        )
    )
    names = [
        node.func.id if isinstance(node, ast.Call) else node.id
        for node in assignment.value.elts
    ]
    assert names.index("ToolNameRepairMiddleware") < names.index(
        "model_fallback_middleware"
    )
