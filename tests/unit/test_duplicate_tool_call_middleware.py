from types import SimpleNamespace

import pytest
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.duplicate_tool_call_middleware import (
    DuplicateToolCallMiddleware,
)


def _request(name, args, call_id, context):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id},
        tool=None,
        state={},
        runtime=SimpleNamespace(context=context, config={"run_id": call_id}),
    )


@pytest.mark.asyncio
async def test_duplicate_tool_call_is_cached_per_run():
    middleware = DuplicateToolCallMiddleware()
    calls = 0
    run_context = {}

    async def handler(request):
        nonlocal calls
        calls += 1
        from langchain_core.messages import ToolMessage

        return ToolMessage(
            content="Link Check Results: 1/1 valid",
            tool_call_id=request.tool_call["id"],
        )

    first = await middleware.awrap_tool_call(
        _request(
            "check_links",
            {"urls": ["https://example.com"]},
            "one",
            run_context,
        ),
        handler,
    )
    repeated = await middleware.awrap_tool_call(
        _request(
            "check_links",
            {"urls": ["https://example.com"]},
            "two",
            run_context,
        ),
        handler,
    )
    separate_run = await middleware.awrap_tool_call(
        _request(
            "check_links",
            {"urls": ["https://example.com"]},
            "three",
            {},
        ),
        handler,
    )

    assert first.content == "Link Check Results: 1/1 valid"
    assert repeated.tool_call_id == "two"
    assert "You already made this exact call" in repeated.content
    assert separate_run.content == "Link Check Results: 1/1 valid"
    assert calls == 2


@pytest.mark.asyncio
async def test_check_links_budget_is_five_calls_per_run():
    middleware = DuplicateToolCallMiddleware()
    calls = 0
    run_context = {}

    async def handler(request):
        nonlocal calls
        calls += 1
        from langchain_core.messages import ToolMessage

        return ToolMessage(
            content=f"result {calls}", tool_call_id=request.tool_call["id"]
        )

    for index in range(5):
        result = await middleware.awrap_tool_call(
            _request(
                "check_links",
                {"urls": [f"https://example.com/{index}"]},
                str(index),
                run_context,
            ),
            handler,
        )
        assert result.content == f"result {index + 1}"

    exhausted = await middleware.awrap_tool_call(
        _request(
            "check_links",
            {"urls": ["https://example.com/5"]},
            "five",
            run_context,
        ),
        handler,
    )

    assert exhausted.content == (
        "Link validation budget exhausted. Write your final answer now using only "
        "URLs already confirmed valid."
    )
    assert calls == 5
