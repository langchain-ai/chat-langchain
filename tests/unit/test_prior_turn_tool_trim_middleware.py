import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.middleware.prior_turn_tool_trim_middleware import PriorTurnToolTrimMiddleware


def _check_links_pair(index: int) -> tuple[AIMessage, ToolMessage]:
    return (
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_links",
                    "args": {"urls": ["https://example.com"]},
                    "id": f"check-{index}",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Link Check Results: 1/1 valid",
            name="check_links",
            tool_call_id=f"check-{index}",
        ),
    )


def _request_with_history() -> ModelRequest:
    messages = [
        SystemMessage(content="You answer questions about documentation."),
        HumanMessage(content="Earlier question"),
        AIMessage(content="Earlier answer"),
    ]
    for index in range(30):
        messages.extend(_check_links_pair(index))
    messages.extend(
        [
            HumanMessage(content="Current question"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_docs",
                        "args": {"query": "current"},
                        "id": "current-search",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Current documentation result",
                name="search_docs",
                tool_call_id="current-search",
            ),
        ]
    )
    return ModelRequest(model=object(), messages=messages)


def test_sync_trims_prior_turn_tool_pairs_and_preserves_current_turn():
    middleware = PriorTurnToolTrimMiddleware()
    request = _request_with_history()
    captured: list[ModelRequest] = []

    def handler(trimmed_request: ModelRequest) -> ModelResponse:
        captured.append(trimmed_request)
        return ModelResponse(result=[AIMessage(content="done")])

    middleware.wrap_model_call(request, handler)

    messages = captured[0].messages
    current_human_index = next(
        index
        for index in range(len(request.messages) - 1, -1, -1)
        if isinstance(request.messages[index], HumanMessage)
    )
    assert messages == [*request.messages[:3], *request.messages[current_human_index:]]
    assert not any(
        isinstance(message, ToolMessage) and message.name == "check_links"
        for message in messages
    )
    assert not any(
        isinstance(message, AIMessage)
        and message.tool_calls
        and message.tool_calls[0]["name"] == "check_links"
        for message in messages
    )


def test_prior_turn_text_ai_with_tool_calls_loses_only_tool_calls():
    middleware = PriorTurnToolTrimMiddleware()
    prior_answer = AIMessage(
        content="The answer is in the documentation.",
        tool_calls=[
            {"name": "check_links", "args": {}, "id": "old", "type": "tool_call"}
        ],
    )
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Old question"),
            prior_answer,
            HumanMessage(content="New question"),
        ],
    )
    captured: list[ModelRequest] = []

    middleware.wrap_model_call(
        request,
        lambda trimmed_request: (
            captured.append(trimmed_request)
            or ModelResponse(result=[AIMessage(content="done")])
        ),
    )

    assert captured[0].messages[1].content == prior_answer.content
    assert captured[0].messages[1].tool_calls == []
    assert captured[0].messages[0] is request.messages[0]
    assert captured[0].messages[-1] is request.messages[-1]


def test_async_matches_sync_and_keeps_current_tool_messages_unchanged():
    middleware = PriorTurnToolTrimMiddleware()
    request = _request_with_history()
    captured: list[ModelRequest] = []

    async def handler(trimmed_request: ModelRequest) -> ModelResponse:
        captured.append(trimmed_request)
        return ModelResponse(result=[AIMessage(content="done")])

    asyncio.run(middleware.awrap_model_call(request, handler))

    current_human_index = next(
        index
        for index in range(len(request.messages) - 1, -1, -1)
        if isinstance(request.messages[index], HumanMessage)
    )
    assert captured[0].messages[-3:] == request.messages[current_human_index:]
    assert captured[0].messages[-2] is request.messages[-2]
    assert captured[0].messages[-1] is request.messages[-1]
    for message in captured[0].messages:
        if isinstance(message, ToolMessage):
            assert any(
                isinstance(previous, AIMessage)
                and any(
                    call["id"] == message.tool_call_id for call in previous.tool_calls
                )
                for previous in captured[0].messages
            )
