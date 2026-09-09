import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.link_check_dedup_middleware import LinkCheckDedupMiddleware


def test_repeated_check_links_call_uses_turn_cache():
    middleware = LinkCheckDedupMiddleware()
    url = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    state = {"messages": [HumanMessage(content="Explain graph APIs.")]}
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=f"Link Check Results: 1/1 valid\n\nValid links:\n  - {url}",
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def run():
        request = ToolCallRequest(
            tool_call={"name": "check_links", "args": {"urls": [url]}, "id": "first"},
            tool=None,
            state=state,
            runtime=None,
        )
        first = await middleware.awrap_tool_call(request, handler)
        second = await middleware.awrap_tool_call(
            request.__class__(
                tool_call={"name": "check_links", "args": {"urls": [url]}, "id": "second"},
                tool=None,
                state=state,
                runtime=None,
            ),
            handler,
        )
        return first, second

    first, second = asyncio.run(run())

    assert calls == 1
    assert "Valid links:" in first.content
    assert "already validated on this turn" in second.content
