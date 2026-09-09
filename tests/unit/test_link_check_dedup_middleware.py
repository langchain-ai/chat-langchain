"""Tests for turn-scoped link check deduplication."""

import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.link_check_dedup_middleware import LinkCheckDedupMiddleware

URL_ONE = "https://docs.langchain.com/one"
URL_TWO = "https://docs.langchain.com/two"


def _request(urls: list[str], turn_id: str = "turn-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "check_links",
            "args": {"urls": urls},
            "id": f"call-{turn_id}",
        },
        tool=None,
        state={"messages": [HumanMessage(content="question", id=turn_id)]},
        runtime=None,
    )


def test_reuses_valid_urls_without_calling_handler_again():
    middleware = LinkCheckDedupMiddleware()
    calls: list[list[str]] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        urls = request.tool_call["args"]["urls"]
        calls.append(urls)
        return ToolMessage(
            content=(
                f"Link Check Results: {len(urls)}/{len(urls)} valid\n\n"
                "Valid links:\n" + "\n".join(f"  - {url}" for url in urls)
            ),
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        first = await middleware.awrap_tool_call(_request([URL_ONE]), handler)
        second = await middleware.awrap_tool_call(_request([URL_ONE]), handler)
        return first, second

    first, second = asyncio.run(invoke())

    assert calls == [[URL_ONE]]
    assert second.content.endswith(
        "(already validated on this turn - do not re-check these URLs)"
    )
    assert URL_ONE in first.content


def test_checks_only_new_urls_in_a_mixed_call():
    middleware = LinkCheckDedupMiddleware()
    calls: list[list[str]] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        urls = request.tool_call["args"]["urls"]
        calls.append(urls)
        return ToolMessage(
            content=(
                f"Link Check Results: {len(urls)}/{len(urls)} valid\n\n"
                "Valid links:\n" + "\n".join(f"  - {url}" for url in urls)
            ),
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        await middleware.awrap_tool_call(_request([URL_ONE]), handler)
        return await middleware.awrap_tool_call(_request([URL_ONE, URL_TWO]), handler)

    result = asyncio.run(invoke())

    assert calls == [[URL_ONE], [URL_TWO]]
    assert URL_ONE in result.content
    assert URL_TWO in result.content
