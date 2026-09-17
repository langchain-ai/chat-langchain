import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.tool_call_name_guard_middleware import ToolCallNameGuardMiddleware


def _request(messages):
    return ModelRequest(
        model=object(),
        messages=messages,
        tools=[{"name": "check_links", "description": "", "parameters": {}}],
    )


def _ai_message(name, *, call_id="call-1", args=None):
    return AIMessage(
        content="",
        response_metadata={"source": "test"},
        tool_calls=[
            {
                "name": name,
                "args": args or {"urls": ["https://example.com"]},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _run_middleware(request):
    calls = []

    async def handler(updated_request):
        calls.append(updated_request)
        return ModelResponse(result=[])

    asyncio.run(ToolCallNameGuardMiddleware().awrap_model_call(request, handler))
    return calls[0]


def test_rewrites_default_api_expression_name():
    request = _request(
        [
            HumanMessage(content="Check links."),
            _ai_message("check_nx<-false if True else default_api:check_links"),
        ]
    )

    updated = _run_middleware(request)

    assert updated.messages[1].tool_calls[0]["name"] == "check_links"


def test_rewrites_long_default_api_expression_name():
    request = _request(
        [
            HumanMessage(content="Check links."),
            _ai_message(
                "check_labs_or_check_links_if_needed_x<-false if True else "
                "default_api:check_links"
            ),
        ]
    )

    updated = _run_middleware(request)

    assert updated.messages[1].tool_calls[0]["name"] == "check_links"


def test_removes_unrecoverable_name_from_both_call_lists():
    request = _request(
        [HumanMessage(content="Check links."), _ai_message("default_api:unknown")]
    )

    updated = _run_middleware(request)
    message = updated.messages[1]

    assert message.tool_calls == []
    assert message.invalid_tool_calls == []


def test_preserves_valid_name_id_args_and_message_metadata():
    message = _ai_message("check_links", call_id="call-7", args={"urls": ["a"]})
    request = _request([HumanMessage(content="Check links."), message])

    updated = _run_middleware(request)
    result = updated.messages[1]

    assert result.tool_calls == message.tool_calls
    assert result.tool_calls[0]["name"] == "check_links"
    assert result.tool_calls[0]["id"] == "call-7"
    assert result.tool_calls[0]["args"] == {"urls": ["a"]}
    assert result.response_metadata == {"source": "test"}
    assert result is message


def test_repairs_malformed_name_in_replayed_history():
    request = _request(
        [
            HumanMessage(content="Earlier request."),
            _ai_message("default_api:check_links", call_id="old-call"),
            HumanMessage(content="Continue from history."),
            AIMessage(content="No tool call here."),
        ]
    )

    updated = _run_middleware(request)

    assert updated.messages[1].tool_calls[0]["name"] == "check_links"
    assert updated.messages[1].tool_calls[0]["id"] == "old-call"
