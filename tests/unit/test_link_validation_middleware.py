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
    expected = f"Read the guide for details.\n- [Valid]({GOOD})\nBare ."
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
@pytest.mark.parametrize("failure", [RuntimeError("failed"), TimeoutError(), "budget"])
async def test_validation_failures_preserve_entire_answer(failure):
    message = AIMessage(content=f"[Bad]({BAD}) [Good]({GOOD})", id="answer")

    async def validate(*args):
        if failure == "budget":
            await asyncio.sleep(1)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("fence, closing", [("```", "```"), ("~~~", "~~~~")])
async def test_code_and_reachable_endpoints_survive_invalid_prose_links(fence, closing):
    docs_mcp = "https://docs.langchain.com/mcp"
    reference_mcp = "https://reference.langchain.com/mcp"
    code = (
        f'{fence}json\r\n{{"servers": [{{"url": "{docs_mcp}"}}, '
        f'{{"url": "{reference_mcp}"}}]}}\r\n{closing}\r\n'
        '```python\nChatOpenAI(base_url="https://api.example.com/v1")\n```\n'
        "```yaml\nurl: http://localhost:2024\n```\n"
        f'Inline `{BAD}` and ``url="{BAD}"``.\n'
    )
    offline = "https://offline.example/endpoint"
    prose = f"[Missing]({BAD}) [Docs]({docs_mcp}) <{reference_mcp}> {offline}"
    message = AIMessage(content=code + prose)

    def respond(request):
        url = str(request.url)
        if url == offline:
            raise httpx.ConnectError("offline", request=request)
        status = 405 if url in (docs_mcp, reference_mcp) else 404
        return httpx.Response(status, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with (
        patch("src.tools.link_check_tools._cache", {}),
        patch("src.tools.link_check_tools.httpx.AsyncClient", return_value=client),
    ):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [message]}, None
        )
    assert result["messages"][0].content == code + (
        f"Missing [Docs]({docs_mcp}) <{reference_mcp}> {offline}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ['URL: "', '- "'])
@pytest.mark.parametrize("suffix", ['"', "'", "}", "]", ")", "."])
async def test_bare_url_removal_preserves_surrounding_delimiters(prefix, suffix):
    message = AIMessage(content=f"{prefix}{BAD}{suffix}")

    async def validate(urls, timeout):
        return [LinkCheckResult(url, False, 404) for url in urls if url == BAD]

    with patch(TARGET, validate):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [message]}, None
        )
    assert result["messages"][0].content == f"{prefix}{suffix}"


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["docs.langchain.com", "example.com"])
@pytest.mark.parametrize("status", [401, 403, 405, 404, 410, 500])
async def test_reachable_http_responses_and_invalid_pages(host, status):
    url = f"https://{host}/status/{status}"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status))
    ) as client:
        with patch("src.tools.link_check_tools._cache", {}):
            result = await _check_single_url(client, url, 1)
    assert result.valid == (status in (401, 403, 405))
    assert result.status_code == status
    assert not result.operational_failure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "heading",
    [
        "Relevant docs:",
        "relevant DOCS",
        "**Relevant docs:**",
        "**Relevant docs**",
        "**Relevant docs**:",
    ],
)
@pytest.mark.parametrize(
    "links", [f"- [Missing]({BAD})\n1. <{BAD}>\n", f"- {BAD}\n", f"<{BAD}>\n"]
)
@pytest.mark.parametrize("block_content", [False, True])
async def test_all_invalid_docs_remove_empty_heading(heading, links, block_content):
    text = f"Complete answer.\n\n{heading}\n\n{links}  \n"
    content = [{"type": "text", "text": text}] if block_content else text
    with patch(TARGET, AsyncMock(return_value=[LinkCheckResult(BAD, False, 404)])):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [AIMessage(content=content)]}, None
        )
    expected = "Complete answer."
    assert result["messages"][0].content == (
        [{"type": "text", "text": expected}] if block_content else expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item", [f"- [Valid]({GOOD})", f"1. [Valid]({GOOD}) with `code`"]
)
async def test_remaining_docs_keep_heading(item):
    text = f"Complete answer.\n\n**Relevant docs:**\n\n- [Missing]({BAD})\n{item}\n"
    with patch(
        TARGET,
        AsyncMock(
            return_value=[
                LinkCheckResult(BAD, False, 404),
                LinkCheckResult(GOOD, True, 200),
            ]
        ),
    ):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [AIMessage(content=text)]}, None
        )
    assert (
        result["messages"][0].content
        == f"Complete answer.\n\n**Relevant docs:**\n\n{item}"
    )


@pytest.mark.asyncio
async def test_empty_heading_cleanup_preserves_sections_and_code():
    code = f"```markdown\nRelevant docs:\n- [Missing]({BAD})\n```\n"
    text = f"Answer with `Relevant docs:`.\n\n{code}\nRelevant docs:\n- [Missing]({BAD})\n\nOther resources:\n- [Valid]({GOOD})\n\n{code}"
    with patch(
        TARGET,
        AsyncMock(
            return_value=[
                LinkCheckResult(BAD, False, 404),
                LinkCheckResult(GOOD, True, 200),
            ]
        ),
    ):
        result = await LinkValidationMiddleware().aafter_model(
            {"messages": [AIMessage(content=text)]}, None
        )
    assert (
        result["messages"][0].content
        == f"Answer with `Relevant docs:`.\n\n{code}\n\nOther resources:\n- [Valid]({GOOD})\n\n{code}"
    )
