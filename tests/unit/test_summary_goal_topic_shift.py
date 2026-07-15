"""Regression tests for topic-shift handling in the context summary goal field."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt


class _CompliantSummaryModel:
    """Fake summary model that follows the goal instruction: goal = latest user message."""

    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt, config=None):
        self.last_prompt = prompt
        latest_user = ""
        for line in prompt.splitlines():
            if line.startswith("Human:"):
                latest_user = line[len("Human:") :].strip()
        goal = (
            f"## Current User Goal\n{latest_user}\n"
            "(potentially stale — re-read the latest user message before "
            "issuing tool calls, and if it diverges from the goal above, "
            "follow the latest message instead)"
        )
        return AIMessage(content=f"Summary of the conversation history until this point:\n\n{goal}")


def _make_middleware(summary_model):
    return CustomSummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([])),
        summary_model=summary_model,
        trigger=("tokens", 130_000),
        keep=("tokens", 30_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    )


def test_goal_prompt_instructs_latest_message_derivation():
    """The goal section must derive from the latest user message and cue staleness."""
    assert "single most recent user message" in context_summary_prompt
    assert "do not carry that older goal forward" in context_summary_prompt
    assert "potentially stale" in context_summary_prompt


def test_pdf_loader_pivot_goal_reflects_latest_message():
    """After a LangSmith-usage thread, a PDF-loader pivot goal must be PDF loaders."""
    model = _CompliantSummaryModel()
    middleware = _make_middleware(model)

    messages = [
        HumanMessage(content="What are the LangSmith usage limits and usage alerts?"),
        AIMessage(content="LangSmith usage limits are configured per workspace..."),
        HumanMessage(content="How do multi-tenancy token quotas work?"),
        AIMessage(content="Token quotas are enforced across the org..."),
        HumanMessage(
            content="please list all PDF loaders and their specific needs in real scenarios"
        ),
    ]

    summary = middleware._create_summary(messages)
    goal_line = summary.split("## Current User Goal", 1)[1].splitlines()[1].lower()

    assert "pdf loader" in goal_line
    assert "usage" not in goal_line
    assert "multi-tenancy" not in goal_line
    assert "token quota" not in goal_line
    assert "potentially stale" in summary.lower()


def test_code_audit_to_tool_vs_skill_pivot_goal_reflects_latest_message():
    """After a code-auditing thread, a tool-vs-skill pivot goal must be tool vs skill."""
    model = _CompliantSummaryModel()
    middleware = _make_middleware(model)

    messages = [
        HumanMessage(content="Audit this code for security vulnerabilities."),
        AIMessage(content="The code has an unsanitized input path..."),
        HumanMessage(content="When should I use a tool versus a skill?"),
    ]

    summary = middleware._create_summary(messages)
    goal_line = summary.split("## Current User Goal", 1)[1].splitlines()[1].lower()

    assert "tool" in goal_line
    assert "skill" in goal_line
    assert "audit" not in goal_line
    assert "vulnerabilit" not in goal_line
