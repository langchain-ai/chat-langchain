"""Exercise pricing routing with a live model and isolated research tools."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools.pricing_tools import fetch_langchain_pricing

pytestmark = pytest.mark.skipif(
    not os.getenv("PRICING_ROUTING_TEST_MODEL"),
    reason="Requires PRICING_ROUTING_TEST_MODEL and its provider credential",
)


@pytest.fixture(params=["instructions", "mirror"])
def system_prompt(request):
    if request.param == "instructions":
        return Path("instructions.md").read_text()
    return docs_agent_prompt


@pytest.fixture
def routing_agent(system_prompt):
    @tool
    def langchainDocs_search_docs_by_lang_chain(query: str) -> str:
        """Search official documentation for page titles and paths."""
        return "LangSmith cost and usage settings: /langsmith/cost-usage.mdx"

    @tool
    def langchainDocs_query_docs_filesystem_docs_by_lang_chain(command: str) -> str:
        """Read official documentation pages found by search."""
        return (
            "LangSmith supports model pricing maps/tables, custom model costs, "
            "cost tracking, usage views, and LLM Gateway credits. "
            "The cost/usage settings example workspace is named cobalt-orchid. "
            "No setting named imaginary_cost_switch is documented here."
        )

    @tool
    def search_support_articles(query: str) -> str:
        """Search support articles."""
        return "No matching articles."

    @tool
    def check_links(urls: list[str]) -> str:
        """Check documentation URLs."""
        return "All URLs valid."

    return create_agent(
        model=init_chat_model(os.environ["PRICING_ROUTING_TEST_MODEL"], temperature=0),
        system_prompt=system_prompt,
        tools=[
            langchainDocs_search_docs_by_lang_chain,
            langchainDocs_query_docs_filesystem_docs_by_lang_chain,
            search_support_articles,
            check_links,
            fetch_langchain_pricing,
        ],
    )


def _run(agent, question, history=()):
    with (
        patch(
            "src.tools.pricing_tools._fetch_pricing_uncached",
            return_value="Current example plan: Plus, 47 USD per seat; 1234 base traces.",
        ),
        patch("src.tools.pricing_tools._cached_text", None),
    ):
        result = agent.invoke(
            {"messages": [*history, HumanMessage(content=question)]},
            config={"recursion_limit": 20},
        )
    messages = result["messages"][len(history) + 1 :]
    tools = [message.name for message in messages if isinstance(message, ToolMessage)]
    return tools, str(messages[-1].content).lower()


@pytest.mark.parametrize(
    "feature",
    [
        "model pricing maps and tables",
        "custom model costs",
        "cost tracking",
        "usage views",
        "LLM Gateway credits",
    ],
)
def test_cost_features_use_documentation(routing_agent, feature):
    tools, answer = _run(
        routing_agent,
        f"Does LangSmith support {feature}? Explain how to use it and "
        "name the example workspace in the documentation.",
    )
    assert "langchainDocs_search_docs_by_lang_chain" in tools
    assert "langchainDocs_query_docs_filesystem_docs_by_lang_chain" in tools
    assert "fetch_langchain_pricing" not in tools
    assert "cobalt-orchid" in answer


@pytest.mark.parametrize(
    "detail",
    [
        "plan tiers",
        "seat prices",
        "invoices",
        "base trace quotas",
        "pay-as-you-go rates",
    ],
)
def test_plan_and_billing_use_live_pricing(routing_agent, detail):
    tools, _ = _run(routing_agent, f"What are LangSmith's current {detail}?")
    assert "fetch_langchain_pricing" in tools


def test_mixed_question_uses_both_sources(routing_agent):
    tools, answer = _run(
        routing_agent,
        "How do custom model costs work, what example workspace do the docs use, "
        "and what is the current Plus seat price?",
    )
    assert "fetch_langchain_pricing" in tools
    assert "langchainDocs_search_docs_by_lang_chain" in tools
    assert "langchainDocs_query_docs_filesystem_docs_by_lang_chain" in tools
    assert "cobalt-orchid" in answer
    assert "47" in answer


def test_feature_denial_requires_current_turn_research(routing_agent):
    history = [
        HumanMessage(content="Does imaginary_cost_switch exist?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "old-search",
                    "name": "langchainDocs_search_docs_by_lang_chain",
                    "args": {"query": "cost"},
                }
            ],
        ),
        ToolMessage(content="No matching pages.", tool_call_id="old-search"),
        AIMessage(content="I could not confirm it in the documentation."),
    ]
    tools, _ = _run(
        routing_agent,
        "Can you confirm that imaginary_cost_switch does not exist?",
        history,
    )
    assert "langchainDocs_search_docs_by_lang_chain" in tools
    assert "langchainDocs_query_docs_filesystem_docs_by_lang_chain" in tools


def test_documented_feature_is_not_denied(routing_agent):
    tools, answer = _run(
        routing_agent,
        "I was told LangSmith has no custom model costs. Is that true? "
        "Check the docs and name their example workspace.",
    )
    assert "langchainDocs_search_docs_by_lang_chain" in tools
    assert "langchainDocs_query_docs_filesystem_docs_by_lang_chain" in tools
    assert "cobalt-orchid" in answer
    for denial in [
        "does not support custom",
        "doesn't support custom",
        "no custom model costs",
    ]:
        assert denial not in answer
