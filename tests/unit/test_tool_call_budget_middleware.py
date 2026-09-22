"""Tests for state-backed per-turn tool-call budgeting."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_call_budget_middleware import ToolCallBudgetMiddleware


def _request(call_id: str):
    return ToolCallRequest(
        tool_call={"name": "search_docs", "id": call_id, "args": {}},
        tool=None,
        state={
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_docs", "args": {}, "id": f"call-{index}"}
                        for index in range(1, 4)
                    ],
                ),
            ]
        },
        runtime=None,
    )


def test_budget_applies_across_separate_async_tasks():
    middleware = ToolCallBudgetMiddleware(max_tool_calls=2)
    executed = []

    async def handler(request):
        executed.append(request.tool_call["id"])
        await asyncio.sleep(0)
        return ToolMessage(
            content="result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        return await asyncio.gather(
            *[
                asyncio.create_task(
                    middleware.awrap_tool_call(_request(f"call-{index}"), handler)
                )
                for index in range(1, 4)
            ]
        )

    results = asyncio.run(invoke())

    assert executed == ["call-1", "call-2"]
    assert results[2].tool_call_id == "call-3"
    assert "budget has been reached" in results[2].content
