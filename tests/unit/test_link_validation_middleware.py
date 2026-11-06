"""Tests for deterministic final-answer link validation."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from src.middleware.link_validation_middleware import LinkValidationMiddleware
from src.tools.link_check_tools import LinkCheckResult, _check_single_url

GOOD = "https://docs.langchain.com/valid"
BAD = "https://docs.langchain.com/missing#details"
TARGET = "src.middleware.link_validation_middleware._check_urls_async"


@pytest.mark.asyncio
@pytest.mark.parametrize("block_content", [False, True])
async def test_mixed_links_preserve_prose_metadata_and_non_text_blocks(block_content):
    text = f"Read [the guide]({BAD}) for details.\n- [Missing]({BAD})\n1. <{BAD}>\n- [Valid]({GOOD})\nBare {BAD}.\n"
    expected = f"Read the guide for details.\n- [Valid]({GOOD})\nBare .\n"
    image = {"type": "image_url", "image_url": {"url": "https://images.example/image"}}
    content = (
        [{"type": "text", "text": text, "extra": "kept"}, image, "More text"]
        if block_content
        else text
    )
    message = AIMessage(
        content=content,
        id="answer",
        response_metadata={"model": "test"},
        additional_kwargs={"custom": True},
    )
    results = [LinkCheckResult(GOOD, True, 200), LinkCheckResult(BAD, False, 404)]
    with patch(TARGET, AsyncMock(return_value=results)):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [message]}, None
        )
    final = add_messages(
        [HumanMessage(content="Question"), message], result["messages"]
    )
    assert len(final) == 2
    answer = final[-1]
    assert answer.content == (
        [{"type": "text", "text": expected, "extra": "kept"}, image, "More text"]
        if block_content
        else expected
    )
    assert answer.id == message.id
    assert answer.response_metadata == message.response_metadata
    assert answer.additional_kwargs == message.additional_kwargs
    assert message.content == content


@pytest.mark.asyncio
async def test_malformed_docs_host_is_removed_without_network():
    url = "https://docs.docs.langchain.com/oss/python#intro"
    message = AIMessage(content=f"See [this guide]({url}).", id="answer")
    result = await LinkValidationMiddleware().aafter_model(
        {"messages": [message]}, None
    )
    assert result["messages"][0].content == "See this guide."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("failed"), TimeoutError(), "budget", "result"]
)
async def test_validation_failures_preserve_entire_answer(failure):
    message = AIMessage(content=f"[Bad]({BAD}) [Good]({GOOD})", id="answer")

    async def validate(*args):
        if failure == "budget":
            await asyncio.sleep(1)
        elif failure == "result":
            return [
                LinkCheckResult(BAD, False, 404),
                LinkCheckResult(
                    GOOD, False, error="Request timed out", operational_failure=True
                ),
            ]
        else:
            raise failure

    with (
        patch(TARGET, validate),
        patch("src.middleware.link_validation_middleware.VALIDATION_BUDGET", 0.01),
    ):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [message]}, None
        )
    assert result is None
    assert message.content == f"[Bad]({BAD}) [Good]({GOOD})"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timeout"),
        httpx.ConnectError("offline"),
        httpx.TooManyRedirects("redirects"),
        RuntimeError("failure"),
    ],
)
async def test_http_failures_are_distinguishable_and_not_cached(error):
    client = AsyncMock()
    client.head.side_effect = error
    url = f"https://example.com/operational-failure/{type(error).__name__}"
    result = await _check_single_url(client, url, 1)
    assert not result.valid
    assert result.operational_failure
    client.head.side_effect = None
    client.head.return_value = httpx.Response(200, request=httpx.Request("HEAD", url))
    recovered = await _check_single_url(client, url, 1)
    assert recovered.valid
    assert not recovered.operational_failure


@pytest.mark.asyncio
async def test_tool_calls_and_non_ai_messages_are_unchanged():
    messages = [
        AIMessage(
            content=BAD, tool_calls=[{"name": "research", "args": {}, "id": "call"}]
        ),
        HumanMessage(content=BAD),
        AIMessage(content="No links"),
    ]
    for message in messages:
        assert (
            await LinkValidationMiddleware().aafter_model({"messages": [message]}, None)
            is None
        )


class ResearchModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_agent_answers_directly_after_research_and_validates_final_message():
    def research(query: str) -> str:
        """Retrieve a source."""
        return f"Source: {GOOD}"

    model = ResearchModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "research", "args": {"query": "docs"}, "id": "research"}
                ],
            ),
            AIMessage(content=f"Read [valid]({GOOD}) and [missing]({BAD})."),
        ]
    )
    agent = create_agent(
        model, tools=[research], middleware=[LinkValidationMiddleware()]
    )
    with patch(
        TARGET,
        AsyncMock(
            return_value=[
                LinkCheckResult(GOOD, True, 200),
                LinkCheckResult(BAD, False, 404),
            ]
        ),
    ):
        result = await agent.ainvoke({"messages": [HumanMessage(content="Find docs")]})
    answers = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(answers) == 2
    assert answers[-1].content == f"Read [valid]({GOOD}) and missing."
    assert [call["name"] for m in answers for call in m.tool_calls] == ["research"]


@pytest.mark.asyncio
async def test_valid_links_leave_answer_unchanged():
    message = AIMessage(content=f"[Guide]({GOOD})", id="answer")
    with patch(TARGET, AsyncMock(return_value=[LinkCheckResult(GOOD, True, 200)])):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [message]}, None
        )
    assert result is None
    assert message.content == f"[Guide]({GOOD})"
