"""Test request-only clearing across turn boundaries."""

import ast
import asyncio
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.middleware.stale_tool_result_middleware import (
    CLEARED_TOOL_RESULT,
    StaleToolResultMiddleware,
)


class RecordingModel(FakeMessagesListChatModel):
    model_inputs: list[list] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.model_inputs.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("content", ["x" * 501, [{"type": "text", "text": "x" * 501}]])
@pytest.mark.parametrize("has_human", [False, True])
def test_request_clearing_preserves_metadata_and_originals(
    asynchronous, content, has_human
):
    old = ToolMessage(
        content=content, tool_call_id="old", name="docs", id="result", status="error"
    )
    current = ToolMessage(content="y" * 1000, tool_call_id="current")
    short = ToolMessage(content="z" * 499, tool_call_id="short")
    boundary = ToolMessage(content="z" * 500, tool_call_id="boundary")
    messages = [
        SystemMessage("System"),
        old,
        short,
        boundary,
        AIMessage("Previous answer"),
    ]
    if has_human:
        messages.append(HumanMessage("Follow up"))
    messages.append(current)
    model = RecordingModel(responses=[AIMessage("Answer")])
    request = ModelRequest(model=model, messages=messages)
    original = [message.model_dump() for message in request.messages]
    response = ModelResponse(result=[AIMessage("Answer")])

    def handler(transformed):
        assert transformed.messages[1].content == (
            CLEARED_TOOL_RESULT if has_human else content
        )
        assert transformed.messages[1].model_dump(
            exclude={"content"}
        ) == old.model_dump(exclude={"content"})
        for i in range(len(messages)):
            if i != 1:
                assert transformed.messages[i] is request.messages[i]
        return response

    async def ahandler(transformed):
        return handler(transformed)

    middleware = StaleToolResultMiddleware()
    result = (
        asyncio.run(middleware.awrap_model_call(request, ahandler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert result is response
    assert [message.model_dump() for message in request.messages] == original
    assert old.content == content


@pytest.mark.parametrize("asynchronous", [False, True])
def test_two_turn_agent_preserves_state_and_current_results(asynchronous):
    payload = "Documentation " * 100

    @tool
    def docs() -> str:
        """Read documentation."""
        return payload

    call = AIMessage("", tool_calls=[{"name": "docs", "args": {}, "id": "lookup"}])
    model = RecordingModel(
        responses=[call, AIMessage("First answer"), AIMessage("Follow-up answer")]
    )
    graph = create_agent(
        model,
        tools=[docs],
        middleware=[StaleToolResultMiddleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "two-turn"}}

    async def run_async():
        first = await graph.ainvoke({"messages": [HumanMessage("Question")]}, config)
        second = await graph.ainvoke({"messages": [HumanMessage("Follow up")]}, config)
        return first, second

    if asynchronous:
        first, second = asyncio.run(run_async())
    else:
        first = graph.invoke({"messages": [HumanMessage("Question")]}, config)
        second = graph.invoke({"messages": [HumanMessage("Follow up")]}, config)
    assert (
        next(m for m in model.model_inputs[1] if isinstance(m, ToolMessage)).content
        == payload
    )
    assert (
        next(m for m in model.model_inputs[2] if isinstance(m, ToolMessage)).content
        == CLEARED_TOOL_RESULT
    )
    assert any(
        isinstance(m, AIMessage) and m.content == "First answer"
        for m in model.model_inputs[2]
    )
    for state in [first, second, graph.get_state(config).values]:
        assert (
            next(m for m in state["messages"] if isinstance(m, ToolMessage)).content
            == payload
        )


def test_clearing_is_wired_after_summarization_before_fallbacks():
    tree = ast.parse(Path("agent.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "docs_agent_middleware"
            for target in node.targets
        )
    )
    names = [
        node.func.id if isinstance(node, ast.Call) else node.id
        for node in assignment.value.elts
    ]
    index = names.index("StaleToolResultMiddleware")
    assert names[index - 1] == "CustomSummarizationMiddleware"
    assert (
        index
        < names.index("model_fallback_middleware")
        < names.index("model_retry_middleware")
    )
