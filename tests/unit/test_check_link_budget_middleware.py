import asyncio

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.check_link_budget_middleware import CheckLinkBudgetMiddleware


def test_check_link_budget_blocks_sixth_call():
    middleware = CheckLinkBudgetMiddleware()
    state = {}
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        return ToolMessage(content="validated", tool_call_id="call")

    async def run_calls():
        results = []
        for index in range(6):
            request = ToolCallRequest(
                tool_call={"name": "check_links", "id": f"call-{index}", "args": {}},
                tool=None,
                state=state,
                runtime=None,
            )
            results.append(await middleware.awrap_tool_call(request, handler))
        return results

    results = asyncio.run(run_calls())

    assert calls == 5
    assert isinstance(results[-1], ToolMessage)
    assert "budget is exhausted" in results[-1].content
