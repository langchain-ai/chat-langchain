import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import push_docs_agent_prompt


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_cross_conversation_memory_limit_is_explicit():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "You cannot remember information across conversations" in instructions
    assert "each new chat starts fresh" in instructions
    assert (
        "If a user asks you to remember something for future conversations"
        in instructions
    )
    assert (
        "tell them plainly you cannot retain it beyond this conversation"
        in instructions
    )
    assert "then offer to help with their LangChain question" in instructions


def test_docs_prompt_publisher_uses_current_instructions(monkeypatch):
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    client = MagicMock()
    monkeypatch.setattr(push_docs_agent_prompt, "Client", lambda: client)
    monkeypatch.setattr(push_docs_agent_prompt, "load_dotenv", MagicMock())

    push_docs_agent_prompt.main()

    client.push_prompt.assert_called_once()
    assert client.push_prompt.call_args.args == (
        "langchain-ai/public-chat-langchain-test",
    )
    prompt = client.push_prompt.call_args.kwargs["object"]
    messages = prompt.format_messages()
    assert len(messages) == 1
    assert messages[0].type == "system"
    assert messages[0].content == instructions


def test_support_search_instructions_match_tool_schema():
    from src.tools.pylon_tools import search_support_articles

    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    schema = search_support_articles.tool_call_schema.model_json_schema()

    assert "query" in schema["properties"]
    assert not schema.get("required")
    assert "`query`" in instructions
    assert "Call `search_support_articles` once with a keyword `query`" in instructions


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="Instruction behavior requires GOOGLE_API_KEY",
)
@pytest.mark.parametrize(
    "identifier", ["import typesafe_sdk", "client = TypeSafeClient()"]
)
@pytest.mark.parametrize("vendor_search_missed", [False, True])
def test_named_integration_research_before_scope_refusal(
    identifier, vendor_search_missed
):
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_core.tools import tool

    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    doc_path = "/oss/python/integrations/chat/typesafe.mdx"

    @tool
    def search_docs_by_lang_chain(query: str) -> str:
        """Search official documentation titles and paths."""
        if "typesafe" in query.lower():
            return f"TypeSafe-compatible models: {doc_path}"
        return "No documentation found."

    @tool
    def query_docs_filesystem_docs_by_lang_chain(command: str) -> str:
        """Read official documentation files with a shell command."""
        if doc_path in command:
            return "TypeSafe integration supports the model ID fixture-typesafe-model."
        return "No matching documentation."

    @tool
    def search_support_articles(query: str, collections: str = "all") -> str:
        """Search support articles by keyword and collection."""
        return "No matching support articles."

    @tool
    def get_support_article_content(article_id: str) -> str:
        """Read a support article returned by search."""
        return "No matching support article."

    messages: list[BaseMessage] = [
        HumanMessage(
            content=(
                "How do I use this integration with Microsoft in LangChain, "
                f"and which model ID does its documentation support?\n```python\n{identifier}\n```"
            )
        )
    ]
    if vendor_search_missed:
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "vendor-search",
                            "name": "search_docs_by_lang_chain",
                            "args": {"query": "microsoft"},
                        }
                    ],
                ),
                ToolMessage(
                    content="No documentation found.", tool_call_id="vendor-search"
                ),
            ]
        )
    agent = create_agent(
        model=init_chat_model("google_genai:gemini-3.5-flash-lite", temperature=0),
        tools=[
            search_docs_by_lang_chain,
            query_docs_filesystem_docs_by_lang_chain,
            search_support_articles,
            get_support_article_content,
        ],
        system_prompt=instructions,
    )
    result = agent.invoke({"messages": messages}, config={"recursion_limit": 16})
    calls = [
        call
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    assert any(
        call["name"] == "search_docs_by_lang_chain"
        and "typesafe" in call["args"]["query"].lower()
        for call in calls
    )
    assert any(
        call["name"] == "query_docs_filesystem_docs_by_lang_chain"
        and doc_path in call["args"]["command"]
        for call in calls
    )
    assert "fixture-typesafe-model" in result["messages"][-1].text
