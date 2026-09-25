import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from src.middleware.retry_middleware import ModelRetryMiddleware
from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def test_model_retry_returns_last_response_after_exhaustion():
    middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    responses = [
        AIMessage(
            content="first",
            response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
        ),
        AIMessage(
            content="last",
            response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
        ),
    ]
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return responses.pop(0)

    result = asyncio.run(middleware.awrap_model_call(SimpleNamespace(), handler))

    assert result.content == "last"
    assert calls == 2


def test_tool_retry_caches_successful_results_across_model_steps():
    middleware = ToolRetryMiddleware(max_attempts=1)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=f"result-{calls}",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    def request(call_id, value):
        return SimpleNamespace(
            tool_call={"name": "search", "args": {"query": value}, "id": call_id},
            runtime=SimpleNamespace(config={"run_id": "root"}, context=None),
        )

    first = asyncio.run(middleware.awrap_tool_call(request("call-1", "same"), handler))
    second = asyncio.run(middleware.awrap_tool_call(request("call-2", "same"), handler))
    different = asyncio.run(
        middleware.awrap_tool_call(request("call-3", "other"), handler)
    )

    assert first.content == "result-1"
    assert second.content == "result-1"
    assert second.tool_call_id == "call-2"
    assert different.content == "result-2"
    assert calls == 2
