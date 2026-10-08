"""Tests for graceful graph recursion exhaustion handling."""

import asyncio
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph

from src.middleware.recursion_boundary import with_recursion_boundary


def test_recursion_error_returns_last_non_empty_assistant_draft():
    def fail(_):
        raise GraphRecursionError("limit reached")

    runnable = with_recursion_boundary(RunnableLambda(fail))
    draft = AIMessage(content="Draft answer")

    result = runnable.invoke({"messages": [HumanMessage(content="Question"), draft]})

    assert result["messages"][-1].content == "Draft answer"


def test_recursion_error_returns_honest_message_without_draft():
    def fail(_):
        raise GraphRecursionError("limit reached")

    runnable = with_recursion_boundary(RunnableLambda(fail))

    result = runnable.invoke({"messages": [HumanMessage(content="Question")]})

    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == (
        "I could not complete this request. Please try again."
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("draft_content", [None, "", "   "])
def test_entrypoint_recursion_error_returns_fallback_in_root_output(
    monkeypatch, caplog, async_mode, draft_content
):
    root_outputs = []
    loop_calls = []

    class RootOutputHandler(BaseCallbackHandler):
        root_run_id = None

        def on_chain_start(
            self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
        ):
            if parent_run_id is None and self.root_run_id is None:
                self.root_run_id = run_id

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            if run_id == self.root_run_id:
                root_outputs.append(outputs)

    def loop(state):
        loop_calls.append(state)
        return {"messages": [AIMessage(content="")]}

    builder = StateGraph(MessagesState)
    builder.add_node("loop", loop)
    builder.add_edge(START, "loop")
    builder.add_edge("loop", "loop")
    compiled_agent = builder.compile()

    monkeypatch.setattr(
        "managed_deepagents.runtime.compile_managed_agent",
        lambda *args, **kwargs: compiled_agent,
    )
    monkeypatch.setitem(
        sys.modules,
        "agent",
        SimpleNamespace(agent=object(), DOCS_AGENT_RECURSION_LIMIT=50),
    )
    monkeypatch.setitem(sys.modules, "_mda_connectors", SimpleNamespace(connectors=[]))
    monkeypatch.setitem(sys.modules, "identity", SimpleNamespace(identity=object()))

    repo_root = Path(__file__).resolve().parents[2]
    entrypoint_path, factory_name = json.loads(
        (repo_root / "langgraph.json").read_text()
    )["graphs"]["docs_agent"].split(":")
    entrypoint = runpy.run_path(str(repo_root / entrypoint_path))
    runnable = entrypoint[factory_name]({})
    messages = [HumanMessage(content="Question")]
    if draft_content is not None:
        messages.append(AIMessage(content=draft_content))
    config = {"callbacks": [RootOutputHandler()]}

    if async_mode:
        result = asyncio.run(runnable.ainvoke({"messages": messages}, config=config))
    else:
        result = runnable.invoke({"messages": messages}, config=config)

    assert root_outputs == [result]
    assert isinstance(root_outputs[-1]["messages"][-1], AIMessage)
    assert root_outputs[-1]["messages"][-1].content == (
        "I could not complete this request. Please try again."
    )
    assert len(loop_calls) == entrypoint["DOCS_AGENT_RECURSION_LIMIT"]
    assert any(
        record.exc_info and isinstance(record.exc_info[1], GraphRecursionError)
        for record in caplog.records
    )
