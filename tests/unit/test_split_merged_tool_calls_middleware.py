from types import SimpleNamespace

from langchain_core.messages import AIMessage

from src.middleware.split_merged_tool_calls_middleware import (
    SplitMergedToolCallsMiddleware,
)


def test_split_merged_tool_calls_restores_missing_call():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "search_docs_by_lang_chainsearch_support_articles",
                "arguments": '{"query": "x"}{"collections": "General"}',
            }
        },
        tool_calls=[
            {
                "name": "search_docs_by_lang_chain",
                "args": {"query": "x"},
                "id": "call_existing",
                "type": "tool_call",
            }
        ],
    )

    update = SplitMergedToolCallsMiddleware().after_model(
        {"messages": [message]}, SimpleNamespace()
    )

    assert update is not None
    assert [call["name"] for call in update["messages"][0].tool_calls] == [
        "search_docs_by_lang_chain",
        "search_support_articles",
    ]
    assert update["messages"][0].tool_calls[1]["args"] == {"collections": "General"}


def test_split_merged_tool_calls_leaves_single_call_unchanged():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "search_docs_by_lang_chain",
                "arguments": '{"query": "x"}',
            }
        },
        tool_calls=[
            {
                "name": "search_docs_by_lang_chain",
                "args": {"query": "x"},
                "id": "call_existing",
                "type": "tool_call",
            }
        ],
    )

    assert (
        SplitMergedToolCallsMiddleware().after_model(
            {"messages": [message]}, SimpleNamespace()
        )
        is None
    )
