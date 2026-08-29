from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.parallel_tool_call_repair_middleware import (
    ParallelToolCallRepairMiddleware,
)


def _request(*tools):
    return SimpleNamespace(tools=list(tools))


def _response(message):
    return ModelResponse(result=[message], structured_response={"answer": "ok"})


def test_repairs_merged_function_call_and_preserves_content():
    message = AIMessage(
        content=[
            {"type": "text", "text": "Researching"},
            {"type": "tool_call", "name": "searchread", "args": {}, "id": "old"},
        ],
        additional_kwargs={
            "function_call": {
                "name": "searchread",
                "arguments": '{"query":"middleware"} {"path":"agents"}',
            }
        },
    )
    response = _response(message)

    result = ParallelToolCallRepairMiddleware().wrap_model_call(
        _request({"name": "search"}, {"function": {"name": "read"}}),
        lambda request: response,
    )

    assert result.structured_response == {"answer": "ok"}
    assert [call["name"] for call in message.tool_calls] == ["search", "read"]
    assert [call["args"] for call in message.tool_calls] == [
        {"query": "middleware"},
        {"path": "agents"},
    ]
    assert message.content[0] == {"type": "text", "text": "Researching"}
    assert message.content[1:] == message.tool_calls
    assert len({call["id"] for call in message.tool_calls}) == 2


@pytest.mark.asyncio
async def test_repairs_async_model_response():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {"name": "ab", "arguments": '{"a":1}{"b":2}'}
        },
    )

    async def handler(request):
        return _response(message)

    await ParallelToolCallRepairMiddleware().awrap_model_call(
        _request({"name": "a"}, {"name": "b"}), handler
    )

    assert [call["name"] for call in message.tool_calls] == ["a", "b"]


def test_ambiguous_segmentation_is_left_unchanged():
    message = AIMessage(
        content="unchanged",
        additional_kwargs={
            "function_call": {"name": "abc", "arguments": '{"a":1}{"b":2}'}
        },
    )
    _response(message)

    ParallelToolCallRepairMiddleware().wrap_model_call(
        _request({"name": "a"}, {"name": "ab"}, {"name": "bc"}, {"name": "c"}),
        lambda request: _response(message),
    )

    assert message.tool_calls == []
    assert message.content == "unchanged"
