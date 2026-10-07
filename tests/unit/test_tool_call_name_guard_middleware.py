"""Tests for malformed tool name repair and history isolation."""

import asyncio
import json
import re

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.middleware import ToolCallNameGuardMiddleware
from src.tools.link_check_tools import check_links


def _request(messages, tools=None):
    return ModelRequest(
        model=object(),
        messages=messages,
        tools=[check_links] if tools is None else tools,
    )


def _message(name, args=None, call_id="call-1"):
    args = {"urls": ["https://docs.langchain.com"]} if args is None else args
    return AIMessage(
        content=[{"type": "tool_use", "id": call_id, "name": name, "input": args}],
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        additional_kwargs={
            "function_call": {"name": name, "arguments": json.dumps(args)},
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
            "provider_metadata": "preserved",
        },
    )


@pytest.mark.parametrize(
    "name",
    [
        "check_classes=",
        "check_classes",
        "check_dates=",
        "check_links|check_links",
        "check_links=",
    ],
)
def test_history_repairs_names_without_mutating_state(name, caplog):
    original = _message(name)
    snapshot = original.model_dump()
    tool_result = ToolMessage(content="valid", tool_call_id="call-1")
    captured = []
    response = ModelResponse(result=[AIMessage(content="answer")])

    async def handler(request):
        captured.append(request)
        return response

    result = asyncio.run(
        ToolCallNameGuardMiddleware().awrap_model_call(
            _request([original, tool_result]), handler
        )
    )

    repaired = captured[0].messages[0]
    assert repaired.tool_calls[0]["name"] == "check_links"
    assert (
        repaired.additional_kwargs["tool_calls"][0]["function"]["name"] == "check_links"
    )
    assert repaired.content[0]["name"] == "check_links"
    assert repaired.additional_kwargs["function_call"]["name"] == "check_links"
    assert repaired.additional_kwargs["provider_metadata"] == "preserved"
    assert captured[0].messages[1] is tool_result
    assert original.model_dump() == snapshot
    assert result is response
    assert name in caplog.text
    assert "message 0" in caplog.text


def test_unrepairable_call_and_matching_result_are_dropped():
    original = _message("unknown=", {"unrecognized": True})
    original.content.insert(0, {"type": "text", "text": "Research"})
    valid = _message("check_links", call_id="call-2")
    messages = [
        original,
        valid,
        ToolMessage(content="bad", tool_call_id="call-1"),
        ToolMessage(content="good", tool_call_id="call-2"),
    ]
    captured = []

    def handler(request):
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="answer")])

    ToolCallNameGuardMiddleware().wrap_model_call(_request(messages), handler)

    cleaned = captured[0].messages
    assert len(cleaned) == 3
    assert cleaned[0].tool_calls == []
    assert "tool_calls" not in cleaned[0].additional_kwargs
    assert "function_call" not in cleaned[0].additional_kwargs
    assert cleaned[0].content == [{"type": "text", "text": "Research"}]
    assert cleaned[1] is valid
    assert cleaned[2] is messages[3]
    assert original.tool_calls[0]["name"] == "unknown="


def test_valid_history_passes_through_unchanged():
    messages = [HumanMessage(content="question"), _message("check_links")]

    def handler(request):
        assert request.messages == messages
        assert all(
            original is cleaned for original, cleaned in zip(messages, request.messages)
        )
        return ModelResponse(result=[AIMessage(content="answer")])

    ToolCallNameGuardMiddleware().wrap_model_call(_request(messages), handler)


@pytest.mark.parametrize("asynchronous", [True, False])
def test_model_response_is_repaired_before_return(asynchronous):
    original = _message("check_classes=")
    original.response_metadata = {"finish_reason": "STOP"}
    response = ModelResponse(result=[original])
    calls = []

    def handler(request):
        calls.append(request)
        return response

    async def async_handler(request):
        return handler(request)

    middleware = ToolCallNameGuardMiddleware()
    request = _request([])
    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert len(calls) == 1
    assert result.result[0].tool_calls[0]["name"] == "check_links"
    assert result.result[0].content[0]["name"] == "check_links"
    assert original.tool_calls[0]["name"] == "check_classes="
    assert response.result[0] is original
    assert result.result[0].response_metadata == {"finish_reason": "STOP"}


@pytest.mark.parametrize("asynchronous", [True, False])
@pytest.mark.parametrize("retry_name", ["check_links", "unrepairable="])
def test_unrepairable_response_retries_once_then_drops(asynchronous, retry_name):
    responses = [
        ModelResponse(result=[_message("unrepairable=", {"unknown": 1})]),
        ModelResponse(result=[_message(retry_name, {"unknown": 1})]),
    ]
    calls = []

    def handler(request):
        calls.append(request)
        return responses[len(calls) - 1]

    async def async_handler(request):
        return handler(request)

    middleware = ToolCallNameGuardMiddleware()
    request = _request([HumanMessage(content="question")])
    result = (
        asyncio.run(middleware.awrap_model_call(request, async_handler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert len(calls) == 2
    assert calls[0] is calls[1]
    assert all(call["name"] == "check_links" for call in result.result[0].tool_calls)
    assert bool(result.result[0].tool_calls) == (retry_name == "check_links")


@pytest.mark.parametrize("nested", [True, False])
def test_dictionary_tool_schemas_and_ambiguous_arguments(nested):
    definition = {
        "name": "check_links",
        "parameters": {"properties": {"urls": {}}, "required": ["urls"]},
    }
    tool = {"type": "function", "function": definition} if nested else definition
    captured = []

    def handler(request):
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="answer")])

    middleware = ToolCallNameGuardMiddleware()
    middleware.wrap_model_call(_request([_message("unknown=")], [tool]), handler)
    assert captured[-1].messages[0].tool_calls[0]["name"] == "check_links"
    other = {**definition, "name": "other_tool"}
    middleware.wrap_model_call(_request([_message("unknown=")], [tool, other]), handler)
    assert captured[-1].messages[0].tool_calls == []


def test_openai_payload_contains_only_provider_valid_names():
    model = ChatOpenAI(api_key="test-key")
    captured = []

    def handler(request):
        captured.append(model._get_request_payload(request.messages))
        return ModelResponse(result=[AIMessage(content="answer")])

    ToolCallNameGuardMiddleware().wrap_model_call(
        _request(
            [
                _message("check_classes="),
                ToolMessage(content="valid", tool_call_id="call-1"),
                _message("check_links|check_links", call_id="call-2"),
                ToolMessage(content="valid", tool_call_id="call-2"),
                _message("unknown=", {"unknown": 1}, call_id="call-3"),
                ToolMessage(content="invalid", tool_call_id="call-3"),
            ]
        ),
        handler,
    )

    names = [
        call["function"]["name"]
        for message in captured[0]["messages"]
        for call in message.get("tool_calls", [])
    ]
    assert names == ["check_links", "check_links"]
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)
    assert "unknown=" not in json.dumps(captured[0])
