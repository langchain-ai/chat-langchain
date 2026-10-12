"""Tests for provider-safe historical tool calls."""

import asyncio
import json
import re

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.middleware.retry_middleware import ModelRetryMiddleware
from src.middleware.tool_name_sanitizer_middleware import ToolNameSanitizerMiddleware


@tool
def check_links(urls: list[str]) -> str:
    """Check documentation links."""
    return "valid"


def _history(name):
    return [
        HumanMessage(content="Check the documentation"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "id": "call-1",
                    "args": {"urls": ["https://docs.example"]},
                }
            ],
        ),
        ToolMessage(content="result", tool_call_id="call-1", name=name),
    ]


def _sanitize(messages, tools, asynchronous):
    request = ModelRequest(model=object(), messages=messages, tools=tools)
    middleware = ToolNameSanitizerMiddleware()
    captured = []
    response = ModelResponse(result=[AIMessage(content="answer")])

    def handler(updated):
        captured.append(updated)
        return response

    async def async_handler(updated):
        return handler(updated)

    if asynchronous:
        result = asyncio.run(middleware.awrap_model_call(request, async_handler))
    else:
        result = middleware.wrap_model_call(request, handler)
    assert result is response
    assert captured[0] is not request
    assert captured[0].messages is not messages
    return captured[0].messages


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "check_links|check_links",
        "check_classes=",
        "check_`links`",
        "check_dates=",
        "check_links->check_links",
        "check_files=",
        "unknown_tool",
        "",
        "check_links\n",
        "a" * 65,
    ],
)
def test_invalid_names_are_sanitized_without_mutating_state(name, asynchronous):
    messages = _history(name)
    snapshot = [message.model_dump() for message in messages]
    sanitized = _sanitize(messages, [check_links], asynchronous)
    call = sanitized[1].tool_calls[0]
    assert call["name"] == "invalid_tool_call"
    assert call["args"] == {"urls": ["https://docs.example"], "_original_name": name}
    assert call["id"] == sanitized[2].tool_call_id == "call-1"
    assert sanitized[2].content == "result"
    assert sanitized[2].name == "invalid_tool_call"
    assert [message.model_dump() for message in messages] == snapshot

    model = ChatOpenAI(model="gpt-5.4-nano", api_key="test", use_responses_api=False)
    payload = model._get_request_payload(sanitized)
    assert (
        payload["messages"][1]["tool_calls"][0]["function"]["name"]
        == "invalid_tool_call"
    )
    assert payload["messages"][2]["tool_call_id"] == "call-1"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema", ["tool", "flat", "openai"])
def test_registered_names_pass_through(schema, asynchronous):
    tools = {
        "tool": check_links,
        "flat": {"name": "check_links"},
        "openai": {"type": "function", "function": {"name": "check_links"}},
    }
    messages = _history("check_links")
    sanitized = _sanitize(messages, [tools[schema]], asynchronous)
    assert sanitized == messages
    assert sanitized[1] is not messages[1]


@pytest.mark.parametrize("name", ["check_links=", "a" * 65])
def test_registered_but_provider_invalid_names_are_sanitized(name):
    sanitized = _sanitize(_history(name), [{"name": name}], False)
    assert sanitized[1].tool_calls[0]["name"] == "invalid_tool_call"


def test_valid_legacy_function_call_is_preserved():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {"name": "check_links", "arguments": '{"urls": []}'}
        },
    )
    assert _sanitize([message], [check_links], False) == [message]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "function_call",
    [
        {"name": "check_links|check_links", "arguments": "{}"},
        {"arguments": "{}"},
        "malformed",
        {"name": "unknown_tool"},
    ],
)
def test_invalid_legacy_function_call_is_removed(function_call, asynchronous):
    message = AIMessage(content="", additional_kwargs={"function_call": function_call})
    sanitized = _sanitize([message], [check_links], asynchronous)
    assert "function_call" not in sanitized[0].additional_kwargs
    assert message.additional_kwargs["function_call"] == function_call


@pytest.mark.parametrize("arguments", ['{"urls": []}', "malformed-json"])
def test_raw_tool_metadata_and_invalid_tool_calls_are_safe(arguments):
    message = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "check_links=", "arguments": arguments},
                }
            ]
        },
    )
    snapshot = message.model_dump()
    sanitized = _sanitize([message], [check_links], False)
    assert (
        sanitized[0].additional_kwargs["tool_calls"][0]["function"]["name"]
        == "invalid_tool_call"
    )
    model = ChatOpenAI(model="gpt-5.4-nano", api_key="test", use_responses_api=False)
    payload = model._get_request_payload(sanitized)
    assert (
        payload["messages"][0]["tool_calls"][0]["function"]["name"]
        == "invalid_tool_call"
    )
    assert message.model_dump() == snapshot


@pytest.mark.parametrize("asynchronous", [False, True])
def test_poisoned_history_reaches_openai_fallback_safely(asynchronous):
    payloads = []

    def respond(request):
        payloads.append((request.url.host, json.loads(request.content)))
        if request.url.host == "primary.example":
            return httpx.Response(
                500, json={"error": {"message": "primary unavailable"}}
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.4-nano",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Documentation answer",
                        },
                    }
                ],
            },
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as client:
        async_client = httpx.AsyncClient(transport=transport)
        models = [
            ChatOpenAI(
                model="gpt-5.4-nano",
                api_key="test",
                max_retries=0,
                base_url=f"https://{host}/v1",
                http_client=client,
                http_async_client=async_client,
                use_responses_api=False,
            )
            for host in ("primary.example", "fallback.example")
        ]
        agent = create_agent(
            models[0],
            tools=[check_links],
            middleware=[
                ToolNameSanitizerMiddleware(),
                *([ModelRetryMiddleware(max_retries=0)] if asynchronous else []),
                ModelFallbackMiddleware(models[1]),
            ],
        )
        messages = _history("check_links|check_links") + _history("check_links")
        messages[4].tool_calls[0]["id"] = "call-2"
        messages[5].tool_call_id = "call-2"
        for index, message in enumerate(messages):
            message.id = f"message-{index}"
        snapshot = [message.model_dump() for message in messages]

        async def invoke():
            try:
                return await agent.ainvoke({"messages": messages})
            finally:
                await async_client.aclose()

        if asynchronous:
            result = asyncio.run(invoke())
        else:
            result = agent.invoke({"messages": messages})
            asyncio.run(async_client.aclose())

    assert result["messages"][-1].content == "Documentation answer"
    assert [host for host, _ in payloads] == ["primary.example", "fallback.example"]
    for _, payload in payloads:
        calls = [
            call
            for message in payload["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert [call["function"]["name"] for call in calls] == [
            "invalid_tool_call",
            "check_links",
        ]
        assert all(
            re.fullmatch(r"[A-Za-z0-9_-]{1,64}", call["function"]["name"])
            for call in calls
        )
        assert [
            message["tool_call_id"]
            for message in payload["messages"]
            if message["role"] == "tool"
        ] == ["call-1", "call-2"]
    assert [message.model_dump() for message in messages] == snapshot
