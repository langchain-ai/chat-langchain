"""Tests for parallel tool-call serialization validation."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from src.middleware.tool_call_validation_middleware import (
    ToolCallNameValidationMiddleware,
)

DECLARED_TOOLS = [
    "search_support_articles",
    "get_support_article_content",
    "fetch_langchain_pricing",
    "check_links",
]

MERGED_NAME = "search_docs_by_lang_chainsearch_support_articles"
MERGED_ARGUMENTS = '{"query": "recursive json splitter"}{"collections": "all"}'


def _middleware() -> ToolCallNameValidationMiddleware:
    return ToolCallNameValidationMiddleware(declared_tools=DECLARED_TOOLS)


def _after_model(middleware, message):
    return middleware.after_model({"messages": [message]}, runtime=SimpleNamespace())


def _serialized_calls(message):
    calls = message.additional_kwargs.get("tool_calls")
    if calls:
        return [call["function"] for call in calls]
    return [message.additional_kwargs["function_call"]]


def test_merged_parallel_function_call_is_split_into_one_call_per_tool(caplog):
    middleware = _middleware()
    message = AIMessage(
        content="",
        id="ai1",
        additional_kwargs={
            "function_call": {"name": MERGED_NAME, "arguments": MERGED_ARGUMENTS}
        },
        tool_calls=[
            {
                "name": "search_docs_by_lang_chain",
                "args": {"query": "recursive json splitter"},
                "id": "tc1",
                "type": "tool_call",
            },
            {
                "name": "search_support_articles",
                "args": {"collections": "all"},
                "id": "tc2",
                "type": "tool_call",
            },
        ],
    )

    with caplog.at_level(logging.WARNING):
        update = _after_model(middleware, message)

    assert update is not None
    assert update["messages"][0].id == "ai1"
    assert MERGED_NAME in caplog.text

    functions = _serialized_calls(message)
    assert [function["name"] for function in functions] == [
        "search_docs_by_lang_chain",
        "search_support_articles",
    ]
    for function in functions:
        assert function["name"] in middleware.declared_tool_names
        parsed = json.loads(function["arguments"])
        assert isinstance(parsed, dict)

    assert [call["name"] for call in message.tool_calls] == [
        "search_docs_by_lang_chain",
        "search_support_articles",
    ]


def test_missing_structured_tool_calls_are_recovered_from_merged_call():
    middleware = _middleware()
    message = AIMessage(
        content="",
        id="ai2",
        additional_kwargs={
            "function_call": {"name": MERGED_NAME, "arguments": MERGED_ARGUMENTS}
        },
    )

    _after_model(middleware, message)

    assert [call["name"] for call in message.tool_calls] == [
        "search_docs_by_lang_chain",
        "search_support_articles",
    ]
    assert message.tool_calls[1]["args"] == {"collections": "all"}


def test_three_way_concatenation_is_detected_and_logged(caplog):
    middleware = _middleware()
    merged = "search_docs_by_lang_chainsearch_docs_by_lang_chainsearch_support_articles"
    message = AIMessage(
        content="",
        id="ai3",
        additional_kwargs={
            "function_call": {
                "name": merged,
                "arguments": '{"query": "a"}{"query": "b"}{"collections": "all"}',
            }
        },
    )

    with caplog.at_level(logging.WARNING):
        _after_model(middleware, message)

    assert merged in caplog.text
    assert len(message.additional_kwargs["tool_calls"]) == 3


def test_unknown_name_is_logged_without_touching_tool_calls(caplog):
    middleware = _middleware()
    tool_calls = [
        {
            "name": "search_support_articles",
            "args": {"collections": "all"},
            "id": "tc1",
            "type": "tool_call",
        }
    ]
    message = AIMessage(
        content="",
        id="ai4",
        additional_kwargs={
            "function_call": {"name": "totally_made_up_tool", "arguments": "{}"}
        },
        tool_calls=list(tool_calls),
    )

    with caplog.at_level(logging.WARNING):
        update = _after_model(middleware, message)

    assert update is None
    assert "totally_made_up_tool" in caplog.text
    assert message.tool_calls == tool_calls
    assert message.additional_kwargs["function_call"]["name"] == "totally_made_up_tool"


def test_well_formed_function_call_is_left_alone():
    middleware = _middleware()
    message = AIMessage(
        content="",
        id="ai5",
        additional_kwargs={
            "function_call": {
                "name": "search_support_articles",
                "arguments": '{"collections": "all"}',
            }
        },
    )

    assert _after_model(middleware, message) is None
    assert "tool_calls" not in message.additional_kwargs


def test_unsplittable_arguments_are_logged_but_not_rewritten(caplog):
    middleware = _middleware()
    message = AIMessage(
        content="",
        id="ai6",
        additional_kwargs={
            "function_call": {
                "name": MERGED_NAME,
                "arguments": '{"query": "recursive json splitter"',
            }
        },
    )

    with caplog.at_level(logging.WARNING):
        update = _after_model(middleware, message)

    assert update is None
    assert MERGED_NAME in caplog.text
    assert message.additional_kwargs["function_call"]["name"] == MERGED_NAME
