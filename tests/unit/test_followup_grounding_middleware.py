"""Tests for per-turn API grounding and bounded research retries."""

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.middleware.followup_grounding_middleware import (
    RETRY_MARKER,
    UNVERIFIED_RESPONSE,
    FollowupGroundingMiddleware,
    extract_api_identifiers,
    successful_content_read,
)

DOCS_READ = "query_docs_filesystem_docs_by_lang_chain"
API = "LLMToolSelectorMiddleware"


def _request(messages):
    return ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=messages,
        state={"messages": messages},
    )


def _read(content=API, name=DOCS_READ, **kwargs):
    return ToolMessage(content=content, name=name, tool_call_id="read", **kwargs)


def _response(content="Unsupported technical answer", tool_name=None):
    calls = [{"name": tool_name, "args": {}, "id": "call"}] if tool_name else []
    return ModelResponse(result=[AIMessage(content=content, tool_calls=calls)])


@pytest.mark.parametrize(
    "identifier",
    [
        "ToolSelectorMiddleware",
        API,
        "HTTPClient",
        "get_support_article_content",
        "create_agent",
        "langchain.agents.create_agent",
        "client.invoke",
    ],
)
def test_extract_api_identifiers(identifier):
    assert extract_api_identifiers(f"How does `{identifier}` work?") == {identifier}


@pytest.mark.parametrize(
    "text",
    [
        "Hello!",
        "Thanks, that helps.",
        "Can you rephrase that?",
        "What is middleware?",
        "Please explain how tools work.",
        "LLM API docs",
    ],
)
def test_ordinary_prose_and_greetings_do_not_qualify(text):
    assert extract_api_identifiers(text) == set()


@pytest.mark.parametrize(
    "message",
    [
        _read(),
        _read(name="get_support_article_content"),
        _read(content=[{"type": "text", "text": API}]),
    ],
)
def test_successful_content_reads(message):
    assert successful_content_read(message)


@pytest.mark.parametrize(
    "message",
    [
        _read(name="search_docs_by_lang_chain"),
        _read(name="search_support_articles"),
        _read(name="check_links"),
        _read(name="unrelated_tool"),
        _read(status="error"),
        _read(content="{}"),
        _read(content="[]"),
        _read(content='{"isError": false, "content": []}'),
        _read(content=""),
        _read(content="Error: unavailable"),
        _read(content='{"error": "Tool unavailable"}'),
        _read(content='{"isError": true, "content": "LLMToolSelectorMiddleware"}'),
        _read(content="No results found."),
        _read(content="head: cannot open page.mdx: No such file or directory"),
        _read(
            content="Article ID missing not found in knowledge base.",
            name="get_support_article_content",
        ),
        _read(
            content="Title: Useful\nContent:\nNo content available",
            name="get_support_article_content",
        ),
        HumanMessage(content=API),
        AIMessage(content=API),
    ],
)
def test_other_output_and_failures_are_not_content_reads(message):
    assert not successful_content_read(message)


def test_new_identifier_retries_with_research_instruction_and_withholds_final():
    request = _request([HumanMessage(content=f"What is {API}?")])
    final = _response()
    handler = AsyncMock(side_effect=[final, final])

    result = asyncio.run(
        FollowupGroundingMiddleware().awrap_model_call(request, handler)
    )

    assert handler.await_count == 2
    retry_request = handler.call_args_list[1].args[0]
    instruction = retry_request.messages[-1]
    assert isinstance(instruction, SystemMessage)
    assert "search_docs_by_lang_chain" in instruction.text
    assert DOCS_READ in instruction.text
    assert API in instruction.text
    assert len(request.messages) == 1
    assert result.result[-1].text == UNVERIFIED_RESPONSE
    assert result.structured_response is None


@pytest.mark.parametrize("name", ["search_docs_by_lang_chain", "check_links"])
@pytest.mark.parametrize("prior", [True, False])
def test_search_and_link_checks_do_not_ground_api(name, prior):
    human = HumanMessage(content=f"What is {API}?")
    messages = [_read(name=name), human] if prior else [human, _read(name=name)]
    handler = AsyncMock(side_effect=[_response(), _response()])

    result = asyncio.run(
        FollowupGroundingMiddleware().awrap_model_call(_request(messages), handler)
    )

    assert handler.await_count == 2
    assert result.result[-1].text == UNVERIFIED_RESPONSE


@pytest.mark.parametrize(
    "messages",
    [
        [HumanMessage(content="Hi!")],
        [HumanMessage(content="Please rephrase that.")],
        [_read(), HumanMessage(content=f"Explain {API} again")],
        [
            _read(name="get_support_article_content"),
            HumanMessage(content=f"Explain {API} again"),
        ],
        [
            HumanMessage(content=f"What is {API}?"),
            _read(content="Relevant page content"),
        ],
        [
            HumanMessage(content=f"What is {API}?"),
            _read(name="get_support_article_content"),
        ],
        [],
    ],
)
def test_grounded_turns_and_non_api_questions_pass_through(messages):
    final = _response()
    handler = AsyncMock(return_value=final)

    result = asyncio.run(
        FollowupGroundingMiddleware().awrap_model_call(_request(messages), handler)
    )

    assert result is final
    assert handler.await_count == 1


def test_all_identifiers_must_be_grounded_by_prior_reads():
    messages = [_read(), HumanMessage(content=f"Compare {API} and OtherMiddleware")]
    handler = AsyncMock(side_effect=[_response(), _response()])

    result = asyncio.run(
        FollowupGroundingMiddleware().awrap_model_call(_request(messages), handler)
    )

    assert result.result[-1].text == UNVERIFIED_RESPONSE
    instruction = handler.call_args_list[1].args[0].messages[-1].text
    assert "OtherMiddleware" in instruction
    assert API not in instruction


@pytest.mark.parametrize("prior", [True, False])
def test_failed_content_reads_do_not_ground_the_turn(prior):
    human = HumanMessage(content=API)
    failed_read = _read(content=f"Error: could not read {API}")
    messages = [failed_read, human] if prior else [human, failed_read]
    handler = AsyncMock(side_effect=[_response(), _response()])

    result = asyncio.run(
        FollowupGroundingMiddleware().awrap_model_call(_request(messages), handler)
    )

    assert handler.await_count == 2
    assert result.result[-1].text == UNVERIFIED_RESPONSE


@pytest.mark.parametrize(
    "response",
    [
        _response(tool_name="search_docs_by_lang_chain"),
        ModelResponse(result=[]),
        ModelResponse(result=[_read()]),
    ],
)
def test_non_final_responses_pass_through(response):
    handler = AsyncMock(return_value=response)

    result = asyncio.run(
        FollowupGroundingMiddleware().awrap_model_call(
            _request([HumanMessage(content=API)]), handler
        )
    )

    assert result is response
    assert handler.await_count == 1


def test_one_retry_per_turn_persists_across_model_and_tool_calls():
    middleware = FollowupGroundingMiddleware()
    messages = [HumanMessage(content=API)]
    search_response = _response(tool_name="search_docs_by_lang_chain")
    handler = AsyncMock(side_effect=[_response(), search_response])
    result = asyncio.run(middleware.awrap_model_call(_request(messages), handler))

    assert result.result[-1].additional_kwargs[RETRY_MARKER]
    assert RETRY_MARKER not in search_response.result[-1].additional_kwargs
    messages.extend([*result.result, _read(name="search_docs_by_lang_chain")])
    handler = AsyncMock(return_value=_response())
    result = asyncio.run(middleware.awrap_model_call(_request(messages), handler))
    assert handler.await_count == 1
    assert result.result[-1].text == UNVERIFIED_RESPONSE

    messages.extend([*result.result, HumanMessage(content="What is OtherMiddleware?")])
    handler = AsyncMock(side_effect=[_response(), _response()])
    asyncio.run(middleware.awrap_model_call(_request(messages), handler))
    assert handler.await_count == 2


def test_retry_budget_does_not_leak_between_threads():
    middleware = FollowupGroundingMiddleware()
    for _ in range(2):
        handler = AsyncMock(side_effect=[_response(), _response()])
        asyncio.run(
            middleware.awrap_model_call(_request([HumanMessage(content=API)]), handler)
        )
        assert handler.await_count == 2


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize(
    "read_content, expected",
    [
        (API + " uses an LLM to select tools.", "Verified technical answer"),
        ("Error: unavailable", UNVERIFIED_RESPONSE),
    ],
)
def test_agent_followup_searches_and_reads_before_technical_answer(
    read_content, expected
):
    calls = []

    @tool
    def search_docs_by_lang_chain(query: str) -> str:
        """Find documentation pages."""
        calls.append("search")
        return "Read /middleware.mdx"

    @tool
    def query_docs_filesystem_docs_by_lang_chain(command: str) -> str:
        """Read documentation content."""
        calls.append("read")
        return read_content

    model = ToolCallingFakeModel(
        responses=[
            AIMessage(content="Unsupported answer"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_docs_by_lang_chain",
                        "args": {"query": API},
                        "id": "search",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DOCS_READ,
                        "args": {"command": "head -200 /middleware.mdx"},
                        "id": "read",
                    }
                ],
            ),
            AIMessage(content="Verified technical answer"),
        ]
    )
    agent = create_agent(
        model,
        tools=[search_docs_by_lang_chain, query_docs_filesystem_docs_by_lang_chain],
        middleware=[FollowupGroundingMiddleware()],
    )
    history = [
        HumanMessage(content="Tell me about tools"),
        _read(content="Tool basics"),
        AIMessage(content="Tools execute functions."),
        HumanMessage(content=f"What about {API}?"),
    ]

    result = asyncio.run(agent.ainvoke({"messages": history}))

    assert calls == ["search", "read"]
    assert result["messages"][-1].text == expected
    assert not any(
        message.text == "Unsupported answer" for message in result["messages"]
    )
    assert (
        sum(
            bool(message.additional_kwargs.get(RETRY_MARKER))
            for message in result["messages"]
            if isinstance(message, AIMessage)
        )
        >= 1
    )


@pytest.mark.parametrize(
    "question, history",
    [
        ("Hello!", []),
        (f"Rephrase the explanation of {API}", [_read()]),
    ],
)
def test_agent_greetings_and_researched_rephrasings_need_no_retry(question, history):
    model = ToolCallingFakeModel(responses=[AIMessage(content="Immediate answer")])
    agent = create_agent(model, middleware=[FollowupGroundingMiddleware()])

    result = asyncio.run(
        agent.ainvoke({"messages": [*history, HumanMessage(content=question)]})
    )

    assert result["messages"][-1].text == "Immediate answer"
    assert not result["messages"][-1].additional_kwargs.get(RETRY_MARKER)


def test_grounding_middleware_is_wired_before_retry_and_fallback():
    tree = ast.parse(Path("agent.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "docs_agent_middleware"
            for target in node.targets
        )
    )
    names = [
        node.func.id if isinstance(node, ast.Call) else node.id
        for node in assignment.value.elts
    ]
    assert names.index("FollowupGroundingMiddleware") < names.index(
        "model_retry_middleware"
    )
    assert names.index("FollowupGroundingMiddleware") < names.index(
        "model_fallback_middleware"
    )
