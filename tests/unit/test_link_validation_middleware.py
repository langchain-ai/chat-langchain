"""Test deterministic final-response link validation and cache reuse."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.link_validation_middleware import LinkValidationMiddleware
from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools import link_check_tools
from src.tools.link_check_tools import LinkCheckResult

GOOD_URL = "https://docs.langchain.com/oss/python/langgraph/use-subgraphs"
BAD_URL = "https://docs.langchain.com/missing"


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(link_check_tools, "_cache", {})


@pytest.fixture
def checker(monkeypatch):
    async def check(urls, timeout):
        return [LinkCheckResult(url=url, valid=url != BAD_URL) for url in urls]

    stub = AsyncMock(side_effect=check)
    monkeypatch.setattr(link_check_tools, "_check_urls_async", stub)
    return stub


def sanitize(message, history=()):
    update = asyncio.run(
        LinkValidationMiddleware().aafter_agent(
            {"messages": [*history, message]}, SimpleNamespace()
        )
    )
    return update["messages"][0] if update else message


def malformed_result(**overrides):
    result = LinkCheckResult(url=BAD_URL, valid=True)
    for name, value in overrides.items():
        setattr(result, name, value)
    return result


def test_1_valid_markdown_and_plain_urls_retained(checker):
    message = AIMessage(content=f"See [Subgraphs]({GOOD_URL}) or {GOOD_URL}.")

    assert sanitize(message) is message
    checker.assert_awaited_once_with([GOOD_URL], link_check_tools.DEFAULT_TIMEOUT)


def test_2_invalid_markdown_links_replaced_with_text(checker):
    message = AIMessage(
        content=f'See [Missing docs]({BAD_URL} "Title") and [Subgraphs]({GOOD_URL}).',
        id="answer",
        response_metadata={"model": "test"},
    )
    result = sanitize(message)

    assert result.content == f"See Missing docs and [Subgraphs]({GOOD_URL})."
    assert result.id == message.id
    assert result.response_metadata == message.response_metadata
    assert BAD_URL in message.content


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (f"Read {BAD_URL} for details.", "Read for details."),
        (
            f"Missing: {BAD_URL}, but keep this sentence.",
            "Missing: , but keep this sentence.",
        ),
        (f"First line\n{BAD_URL}\nLast line", "First line\n\nLast line"),
        (f"See <{BAD_URL}> for details.", "See for details."),
        (BAD_URL, ""),
    ],
)
def test_3_invalid_plain_urls_removed_without_losing_prose(checker, content, expected):
    assert sanitize(AIMessage(content=content)).content == expected


def test_4_cached_urls_not_checked_again(checker):
    link_check_tools._cache.update(
        {
            GOOD_URL: LinkCheckResult(url=GOOD_URL, valid=True),
            BAD_URL: LinkCheckResult(url=BAD_URL, valid=False),
        }
    )
    result = sanitize(AIMessage(content=f"[Good]({GOOD_URL}) [Bad]({BAD_URL})"))

    assert result.content == f"[Good]({GOOD_URL}) Bad"
    checker.assert_not_awaited()


def test_5_tool_calling_and_intermediate_messages_unchanged(checker):
    intermediate = AIMessage(content=BAD_URL, id="intermediate")
    tool_calling = AIMessage(
        content=BAD_URL,
        tool_calls=[
            {"name": "search", "args": {}, "id": "call"},
        ],
    )
    tool_result = ToolMessage(content=BAD_URL, tool_call_id="call")
    final = AIMessage(content="Final answer")

    assert sanitize(tool_calling, [intermediate]) is tool_calling
    assert sanitize(tool_result, [intermediate, tool_calling]) is tool_result
    assert sanitize(final, [intermediate, tool_calling, tool_result]) is final
    assert intermediate.content == BAD_URL
    checker.assert_not_awaited()


@pytest.mark.parametrize(
    "outcome",
    [
        RuntimeError("checker unavailable"),
        None,
        {},
        [],
        [None, {"url": BAD_URL, "valid": True}],
        [malformed_result(valid="yes")],
        [LinkCheckResult(url=GOOD_URL, valid=True)],
        [malformed_result(url=[])],
    ],
)
def test_6_failures_and_malformed_results_fail_closed(monkeypatch, outcome):
    checker = AsyncMock(
        side_effect=outcome if isinstance(outcome, Exception) else None,
        return_value=outcome,
    )
    monkeypatch.setattr(link_check_tools, "_check_urls_async", checker)
    result = sanitize(AIMessage(content=f"[Missing]({BAD_URL}) {BAD_URL}"))

    assert result.content == "Missing "
    assert BAD_URL not in result.content


def test_6_malformed_cached_results_not_treated_as_valid(checker):
    link_check_tools._cache[BAD_URL] = {"url": BAD_URL, "valid": True}

    assert sanitize(AIMessage(content=BAD_URL)).content == ""
    checker.assert_not_awaited()


def test_content_blocks_preserve_non_text_and_metadata(checker):
    content = [
        {"type": "text", "text": f"[Missing]({BAD_URL})", "extra": "keep"},
        GOOD_URL,
        {"type": "reasoning", "reasoning": BAD_URL},
    ]
    result = sanitize(AIMessage(content=content))

    assert result.content == [
        {"type": "text", "text": "Missing", "extra": "keep"},
        GOOD_URL,
        content[-1],
    ]


def test_balanced_parentheses_and_markdown_titles(checker):
    url = "https://example.com/path_(example)"
    content = f'See [Example]({url} "Title") or ({url}).'

    assert sanitize(AIMessage(content=content)).content == content
    checker.assert_awaited_once_with([url], link_check_tools.DEFAULT_TIMEOUT)


def test_prompts_make_checks_optional_and_automatic():
    for prompt in (docs_agent_prompt, Path("instructions.md").read_text()):
        assert "Links in your final response are validated automatically" in prompt
        assert (
            "Optionally call `check_links` for constructed anchor links not yet validated"
            in prompt
        )
        assert "Before finalizing your response, call `check_links`" not in prompt
        assert "Call `check_links` with the URLs you plan to include" not in prompt
        assert "Called `check_links`" not in prompt


def test_docs_agent_registers_link_validation():
    from agent import docs_agent_middleware

    assert any(
        isinstance(item, LinkValidationMiddleware) for item in docs_agent_middleware
    )


def test_follow_up_graph_reuses_checks_without_model_tool_calls(checker):
    class DocsResponseModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = DocsResponseModel(
        responses=[
            AIMessage(content=f"[Subgraphs]({GOOD_URL}) [Missing]({BAD_URL})"),
            AIMessage(content=f"Again: [Subgraphs]({GOOD_URL}) {BAD_URL}"),
        ]
    )
    graph = create_agent(
        model=model,
        tools=[link_check_tools.check_links],
        middleware=[LinkValidationMiddleware()],
    )

    async def conversation():
        first = await graph.ainvoke({"messages": [HumanMessage(content="Subgraphs?")]})
        second = await graph.ainvoke(
            {
                "messages": [
                    *first["messages"],
                    HumanMessage(content="Follow up"),
                ]
            }
        )
        return first, second

    first, second = asyncio.run(conversation())

    assert first["messages"][-1].content == f"[Subgraphs]({GOOD_URL}) Missing"
    assert second["messages"][-1].content == f"Again: [Subgraphs]({GOOD_URL}) "
    assert not any(
        message.tool_calls
        for message in second["messages"]
        if isinstance(message, AIMessage)
    )
    checker.assert_awaited_once_with(
        [GOOD_URL, BAD_URL], link_check_tools.DEFAULT_TIMEOUT
    )
