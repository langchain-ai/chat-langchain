"""Tests for deterministic final-response link validation."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from src.middleware.link_validation_middleware import LinkValidationMiddleware
from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools import link_check_tools
from src.tools.link_check_tools import LinkCheckResult

GOOD = "https://docs.langchain.com/good"
BAD = "https://docs.langchain.com/missing"


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.setattr(link_check_tools, "_cache", {})
    mock = AsyncMock(
        side_effect=lambda urls, timeout: [
            LinkCheckResult(url=url, valid=url != BAD) for url in urls
        ]
    )
    monkeypatch.setattr(link_check_tools, "_check_urls_async", mock)
    return mock


def sanitize(content):
    message = AIMessage(content=content, id="final")
    update = LinkValidationMiddleware().after_agent(
        {"messages": [HumanMessage(content="Question"), message]}, Runtime()
    )
    if update is None:
        return content
    assert update["messages"][0].id == "final"
    assert message.content == content
    return update["messages"][0].content


def test_valid_markdown_and_plain_links_retained(checker):
    text = f"[Docs]({GOOD}) and {GOOD}."
    assert sanitize(text) == text
    checker.assert_awaited_once_with([GOOD], link_check_tools.DEFAULT_TIMEOUT)


def test_invalid_markdown_links_become_link_text(checker):
    assert (
        sanitize(f'Read [the docs]({BAD} "Docs title") first.')
        == "Read the docs first."
    )
    assert sanitize(f"[**Docs**](<{BAD}>)") == "**Docs**"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"Read {BAD} for details.", "Read for details."),
        (f"Read {BAD}.", "Read."),
        (f"Read {BAD} {BAD}. Next.", "Read. Next."),
        (f"See ({BAD}). Next.", "See. Next."),
        (f"See <{BAD}> for details.", "See for details."),
        (f"{BAD}\nNext paragraph.", "\nNext paragraph."),
        (BAD, ""),
    ],
)
def test_invalid_plain_urls_removed_without_damaging_prose(checker, text, expected):
    assert sanitize(text) == expected


def test_balanced_parentheses_and_punctuation_not_part_of_plain_url(checker):
    url = "https://example.com/path_(details)"
    text = f"See ({url}), then [Docs]({url})."
    assert sanitize(text) == text
    checker.assert_awaited_once_with([url], link_check_tools.DEFAULT_TIMEOUT)


def test_cached_links_not_checked_again(checker):
    link_check_tools._cache[GOOD] = LinkCheckResult(url=GOOD, valid=True)
    link_check_tools._cache[BAD] = LinkCheckResult(url=BAD, valid=False)
    assert sanitize(f"[Good]({GOOD}) and [Bad]({BAD})") == f"[Good]({GOOD}) and Bad"
    checker.assert_not_awaited()


def test_only_uncached_unique_urls_checked(checker):
    link_check_tools._cache[GOOD] = LinkCheckResult(url=GOOD, valid=True)
    assert sanitize(f"[Good]({GOOD}) {BAD} {BAD}") == f"[Good]({GOOD}) "
    checker.assert_awaited_once_with([BAD], link_check_tools.DEFAULT_TIMEOUT)


def test_tool_calls_and_intermediate_messages_unchanged(checker):
    intermediate = AIMessage(content=BAD, id="intermediate")
    tool_call = AIMessage(
        content=BAD,
        tool_calls=[{"id": "call", "name": "search", "args": {}}],
    )
    middleware = LinkValidationMiddleware()
    for messages in [
        [],
        [intermediate, HumanMessage(content="Follow-up")],
        [tool_call],
        [tool_call, ToolMessage(content=BAD, tool_call_id="call")],
    ]:
        assert middleware.after_agent({"messages": messages}, Runtime()) is None
    final = AIMessage(content=f"[Missing]({BAD})", id="final")
    update = middleware.after_agent({"messages": [intermediate, final]}, Runtime())
    assert update["messages"][0].content == "Missing"
    assert intermediate.content == BAD


def test_checker_failure_removes_unverified_links_but_keeps_cached_valid(checker):
    link_check_tools._cache[GOOD] = LinkCheckResult(url=GOOD, valid=True)
    checker.side_effect = RuntimeError("Checker unavailable")
    assert sanitize(f"[Good]({GOOD}) [Missing]({BAD})") == f"[Good]({GOOD}) Missing"


@pytest.mark.parametrize(
    "results",
    [
        None,
        {},
        [None],
        [{"url": BAD, "valid": True}],
        [],
        [LinkCheckResult(url=GOOD, valid=True)],
        [LinkCheckResult(url=BAD, valid="yes")],
    ],
)
def test_malformed_results_do_not_validate_unchecked_urls(checker, results):
    checker.side_effect = None
    checker.return_value = results
    assert sanitize(f"[Missing]({BAD}) {BAD}") == "Missing "


def test_malformed_cache_entry_is_not_treated_as_valid(checker):
    link_check_tools._cache[BAD] = {"url": BAD, "valid": True}
    assert sanitize(f"[Missing]({BAD})") == "Missing"
    checker.assert_not_awaited()


def test_text_blocks_sanitized_and_nontext_blocks_preserved(checker):
    content = [
        {"type": "text", "text": f"[Missing]({BAD})"},
        {"type": "image_url", "image_url": {"url": GOOD}},
        f"More at {GOOD}.",
    ]
    assert sanitize(content) == [
        {"type": "text", "text": "Missing"},
        content[1],
        content[2],
    ]


class DocsModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_followup_reuses_validation_without_model_tool_calls(checker):
    async def run():
        model = DocsModel(
            responses=[
                AIMessage(content=f"[Docs]({GOOD}) and [Missing]({BAD})"),
                AIMessage(content=f"Again: [Docs]({GOOD}) and {BAD}."),
                AIMessage(content="Unexpected extra model call"),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[link_check_tools.check_links],
            system_prompt=docs_agent_prompt,
            middleware=[LinkValidationMiddleware()],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "followup"}}
        for question in ["Question", "Follow-up"]:
            events = [
                event
                async for event in agent.astream(
                    {"messages": [HumanMessage(content=question)]},
                    config,
                    stream_mode=["messages", "values"],
                )
            ]
            final_state = [data for mode, data in events if mode == "values"][-1]
            assert BAD not in final_state["messages"][-1].content
            assert GOOD in final_state["messages"][-1].content
            assert not any(
                isinstance(msg, ToolMessage) for msg in final_state["messages"]
            )
        assert model.i == 2

    asyncio.run(run())
    checker.assert_awaited_once_with([GOOD, BAD], link_check_tools.DEFAULT_TIMEOUT)


def test_docs_agent_registers_link_validation_middleware(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_PROMPTS", "1")
    for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.setenv(key, "test-key")
    from agent import docs_agent_middleware

    assert any(
        isinstance(middleware, LinkValidationMiddleware)
        for middleware in docs_agent_middleware
    )


def test_prompt_copies_make_pre_response_link_checks_optional():
    instructions = Path("instructions.md").read_text()
    for prompt in [docs_agent_prompt, instructions]:
        assert "Links in the final response are validated automatically" in prompt
        assert (
            "Optionally call `check_links` for constructed anchor links not yet validated"
            in prompt
        )
        assert "Validate links BEFORE sending" not in prompt
        assert "Before finalizing your response, call `check_links`" not in prompt
        assert "Called `check_links` to verify URLs work" not in prompt
