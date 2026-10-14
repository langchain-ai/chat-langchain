"""Tests for request-time dates and the docs agent's recency prompt contract."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts.docs_agent_prompt import docs_agent_prompt

INSTRUCTIONS = (Path(__file__).resolve().parents[2] / "instructions.md").read_text()


@pytest.fixture
def entrypoint(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_PROMPTS", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "test-key")

    import agent

    return agent


@pytest.fixture
def clock(monkeypatch, entrypoint):
    class Clock:
        current = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz):
            assert tz is timezone.utc
            return cls.current

    monkeypatch.setattr(entrypoint, "datetime", Clock)
    return Clock


def test_current_date_prompt_refreshes_and_preserves_system_message(clock, entrypoint):
    system_message = SystemMessage(
        content=[{"type": "text", "text": INSTRUCTIONS}], id="instructions"
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        messages=[],
        system_message=system_message,
    )
    prompts = []

    def handler(updated_request):
        prompts.append(updated_request.system_message)
        return ModelResponse(result=[AIMessage(content="unused")])

    entrypoint.current_date_prompt.wrap_model_call(request, handler)
    clock.current += timedelta(minutes=2)
    entrypoint.current_date_prompt.wrap_model_call(request, handler)

    assert prompts[0].content_blocks[-1]["text"] == "Current UTC date: 2026-01-31"
    assert prompts[1].content_blocks[-1]["text"] == "Current UTC date: 2026-02-01"
    assert all(prompt.content_blocks[0]["text"] == INSTRUCTIONS for prompt in prompts)
    assert all(prompt.id == "instructions" for prompt in prompts)
    assert system_message.content == [{"type": "text", "text": INSTRUCTIONS}]
    assert entrypoint.current_date_prompt in entrypoint.docs_agent_middleware


@pytest.mark.asyncio
async def test_current_date_prompt_async_without_existing_system_message(
    clock, entrypoint
):
    request = ModelRequest(model=FakeListChatModel(responses=["unused"]), messages=[])
    prompts = []

    async def handler(updated_request):
        prompts.append(updated_request.system_message)
        return ModelResponse(result=[AIMessage(content="unused")])

    await entrypoint.current_date_prompt.awrap_model_call(request, handler)

    assert prompts[0].text == "Current UTC date: 2026-01-31"


@pytest.mark.asyncio
async def test_agent_model_receives_date_and_time_relative_guidance(clock, entrypoint):
    prompts = []

    class RecordingModel(FakeListChatModel):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            prompts.append(messages[0])
            return super()._call(messages, stop, run_manager, **kwargs)

    graph = create_agent(
        model=RecordingModel(responses=["unused"]),
        system_prompt=INSTRUCTIONS,
        middleware=[entrypoint.current_date_prompt],
    )
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="What changed in Python in the last 30 days?")
            ]
        }
    )

    assert "Current UTC date: 2026-01-31" in prompts[0].text
    assert "### Time-relative questions" in prompts[0].text
    assert INSTRUCTIONS in prompts[0].text


def test_time_relative_guidance_is_synchronized():
    def section(prompt):
        return prompt.split("### Time-relative questions\n", 1)[1].split(
            "**Default mode:", 1
        )[0]

    assert section(INSTRUCTIONS) == section(docs_agent_prompt)


@pytest.mark.parametrize(
    "required_guidance",
    [
        (
            "dated changelog entry falls within that window and is not in the future",
            "cite its release date (YYYY-MM-DD) and changelog source for each qualifying change",
        ),
        (
            "no dated changelog entry falls within the requested window",
            "explicitly say that no qualifying documented changes were found",
        ),
        (
            "the changelog is empty",
            "explicitly say that no qualifying documented changes were found",
        ),
        (
            "Select the Python or JavaScript changelog using the user's language and product context; they can differ",
            "Use product-specific changelogs for LangGraph, LangSmith, Fleet, or Deep Agents",
        ),
        (
            "Undated reference material can explain existing features",
            "must not be described as new or used as evidence of a recent change",
        ),
    ],
    ids=["in-window", "out-of-window", "empty", "language-product", "undated"],
)
def test_time_relative_answer_requirements(required_guidance):
    for prompt in (INSTRUCTIONS, docs_agent_prompt):
        for requirement in required_guidance:
            assert requirement in prompt
