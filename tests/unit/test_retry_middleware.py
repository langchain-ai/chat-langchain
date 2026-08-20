import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.retry_middleware import ModelRetryMiddleware

TOOL_NAMES = [
    "get_support_article_content",
    "search_support_articles",
    "search_docs_by_lang_chain",
]


def _request() -> ModelRequest:
    return ModelRequest(
        model=object(),
        messages=[],
        tools=[{"name": tool_name} for tool_name in TOOL_NAMES],
    )


def _response(name: str, arguments: str, tool_calls: list[dict]) -> ModelResponse:
    return ModelResponse(
        result=[
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": name, "arguments": arguments}
                },
                tool_calls=tool_calls,
            )
        ]
    )


def test_merged_function_call_repairs_two_tool_calls():
    response = _response(
        "search_docs_by_lang_chainsearch_support_articles",
        '{"query": "x"}{"collections": "all"}',
        [
            {
                "name": "search_support_articles",
                "args": {"collections": "all"},
                "id": "old",
                "type": "tool_call",
            }
        ],
    )

    result = asyncio.run(
        ModelRetryMiddleware(max_retries=0).awrap_model_call(
            _request(), lambda _: _return(response)
        )
    )
    message = result.result[0]

    assert [tool_call["name"] for tool_call in message.tool_calls] == [
        "search_docs_by_lang_chain",
        "search_support_articles",
    ]
    assert [tool_call["args"] for tool_call in message.tool_calls] == [
        {"query": "x"},
        {"collections": "all"},
    ]
    assert len({tool_call["id"] for tool_call in message.tool_calls}) == 2
    assert "function_call" not in message.additional_kwargs


def test_merged_function_call_repairs_three_tool_calls():
    response = _response(
        "search_docs_by_lang_chainsearch_support_articlesget_support_article_content",
        '{"query": "x"}{"collections": "all"}{"article_id": "1"}',
        [],
    )

    result = asyncio.run(
        ModelRetryMiddleware(max_retries=0).awrap_model_call(
            _request(), lambda _: _return(response)
        )
    )

    assert [tool_call["name"] for tool_call in result.result[0].tool_calls] == [
        "search_docs_by_lang_chain",
        "search_support_articles",
        "get_support_article_content",
    ]


def test_legitimate_function_call_is_untouched():
    response = _response(
        "search_support_articles",
        '{"collections": "all"}',
        [
            {
                "name": "search_support_articles",
                "args": {"collections": "all"},
                "id": "existing",
                "type": "tool_call",
            }
        ],
    )
    original = response.result[0].additional_kwargs.copy()

    result = asyncio.run(
        ModelRetryMiddleware(max_retries=0).awrap_model_call(
            _request(), lambda _: _return(response)
        )
    )

    assert result.result[0].tool_calls[0]["id"] == "existing"
    assert result.result[0].additional_kwargs == original


def test_unknown_function_call_is_untouched():
    response = _response(
        "unknown_toolsearch_support_articles",
        '{"query": "x"}{"collections": "all"}',
        [],
    )

    result = asyncio.run(
        ModelRetryMiddleware(max_retries=0).awrap_model_call(
            _request(), lambda _: _return(response)
        )
    )

    assert result.result[0].tool_calls == []
    assert result.result[0].additional_kwargs["function_call"]["name"] == (
        "unknown_toolsearch_support_articles"
    )


async def _return(response: ModelResponse) -> ModelResponse:
    return response
